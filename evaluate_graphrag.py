"""Ý 2: GraphRAG vs Vector RAG trên cùng 30 bài báo — retrieval precision + generation quality.

    python evaluate_graphrag.py --run kg30                 # cả 2 hệ, mọi câu hỏi thuộc 30 bài
    python evaluate_graphrag.py --run thu --limit 5        # chạy thử

30 bài chủ đề quantum (data/kg/papers.json). Hai hệ (cùng 30 bài, cùng chunk, cùng TOP_K đoạn văn, cùng prompt +
LLM sinh câu trả lời):
  - vector_rag : LangChain RAG của ý 1 (rag_langchain.py, embedding chunk), trên Chroma riêng của ý 2 chứa đúng
                 chunk của 30 bài (kg_build.use_kg_store; data/kg/chroma.zip)
  - graphrag   : rag_graphrag.py, truy xuất qua KG (entity -> quan hệ -> chunk) + đưa mô tả entity/quan hệ vào context

Bộ câu hỏi: các câu của ý 1 có nguồn nằm trong 30 bài (single-hop, multi-hop) + câu sinh riêng cho ý 2
(testset/singlehop_kg.json cho 7 bài bổ sung, testset/multihop_kg.json; đã duyệt tay) + câu ngoài corpus
(đúng = từ chối).

Retrieval precision (trên TOP_K đoạn văn mỗi hệ lấy ra):
  - context_precision  = #đoạn hữu ích / TOP_K; judge quyết định mỗi đoạn có giúp suy ra đáp án chuẩn không
                         (RAGAS context precision, bản không trọng số thứ hạng) -> CHỈ SỐ CHÍNH
  - context_ap (phụ)   = average precision theo thứ hạng (đúng công thức RAGAS context_precision)
  - source_precision   = tỉ lệ đoạn thuộc bài nguồn của câu hỏi (không cần judge)
  - retrieval_hit / gold_chunk_recall: có đủ bài nguồn / tỉ lệ đoạn gốc (đoạn dùng để sinh câu hỏi) được lấy
  - context_recall     = #claim của đáp án chuẩn suy ra được từ context / #claim (context = KG + đoạn văn)
Generation quality: faithfulness, answer_relevancy, hallucination_rate (như ý 1) + answer_correctness.
Faithfulness chấm trên đúng context mà hệ đưa vào LLM (GraphRAG: mô tả KG + đoạn văn); cột *_text (phụ) chỉ
chấm trên đoạn văn -> cho biết bao nhiêu claim chỉ dựa vào mô tả do LLM trích ra trong KG.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

import config
from common import env_info, is_refusal
# Dùng lại nguyên prompt judge / hàm chấm của ý 1 (evaluate.py không bị sửa)
from evaluate import (CORRECTNESS, EXTRACT_CLAIMS, REVERSE_QUESTIONS, TITLES, append_jsonl, bootstrap_ci, cosine,
                      judge, load_jsonl, load_testset, verify)
from common import get_embeddings
from kg_build import load_papers, use_kg_store

SYSTEMS = {"vector_rag": "rag_langchain", "graphrag": "rag_graphrag"}
TESTSET_FILES = [config.TESTSET_FILE, *(config.TESTSET_FILE.parent / f for f in
                                        ("singlehop_kg.json", "multihop.json", "multihop_kg.json", "unanswerable.json"))]
TYPES = ("single", "multihop", "unanswerable")

USEFUL = ChatPromptTemplate.from_messages([
    ("system", "You judge retrieved passages for a question that has a reference answer. For each numbered "
               "passage, decide whether it contains information that helps to arrive at the reference answer "
               "('yes') or not ('no'). Passages that are only on the same topic but do not contain facts of the "
               "reference answer are 'no'. "
               'Return JSON: {{"verdicts": [{{"id": 1, "useful": "yes" or "no"}}, ...]}} '
               "with one entry per passage."),
    ("human", "Question: {question}\nReference answer: {reference}\n\nPassages:\n{passages}"),
])


def extract_claims(question: str, answer: str) -> list[str]:
    return [c for c in judge(EXTRACT_CLAIMS, question=question, answer=answer).get("claims", [])
            if isinstance(c, str) and c.strip()] or [answer]


def answer_relevancy(question: str, answer: str) -> float:
    """Như ý 1: trung bình cosine(câu hỏi gốc, 3 câu hỏi sinh ngược từ câu trả lời); 0 nếu né tránh."""
    r = judge(REVERSE_QUESTIONS, answer=answer)
    gen_qs = [x for x in r.get("questions", []) if isinstance(x, str) and x.strip()][:3]
    if r.get("noncommittal") is True or not gen_qs:
        return 0.0
    emb = get_embeddings()
    qv, gvs = emb.embed_query(question), emb.embed_documents(gen_qs)
    return float(np.mean([cosine(qv, g) for g in gvs]))


def answer_correctness(item: dict, answer: str, refusal: bool) -> int:
    """Như ý 1: so với đáp án chuẩn; câu ngoài corpus thì đúng = từ chối."""
    if item["type"] == "unanswerable":
        return int(refusal)
    return 0 if refusal else int(judge(CORRECTNESS, question=item["question"], reference=item["ground_truth"],
                                       answer=answer).get("score", 0) in (1, "1"))


def useful_flags(item: dict, texts: list[str]) -> list[int]:
    if not texts:
        return []
    passages = "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    verdicts = judge(USEFUL, question=item["question"], reference=item["ground_truth"],
                     passages=passages).get("verdicts", [])
    ok = {int(v.get("id", 0)): str(v.get("useful", "")).lower().startswith("y")
          for v in verdicts if isinstance(v, dict) and str(v.get("id", "")).isdigit()}
    return [int(ok.get(i + 1, False)) for i in range(len(texts))]


def average_precision(flags: list[int]) -> float:
    hits, total = 0, 0.0
    for i, f in enumerate(flags):
        if f:
            hits += 1
            total += hits / (i + 1)
    return total / hits if hits else 0.0


def gold_sources(item: dict) -> list[str]:
    return item.get("sources") or [item["source"]]


def gold_chunks(item: dict) -> list[str]:
    return item.get("chunk_ids") or [item["chunk_id"]]


def score(item: dict, run: dict) -> dict:
    q, ans = item["question"], run["answer"]
    texts = run["contexts"]
    full = texts + ([{"text": run["kg_context"]}] if run.get("kg_context") else [])
    refusal = is_refusal(ans)
    s = {"refusal": int(refusal), "n_chunks": len(texts)}

    # Retrieval (chỉ câu có đáp án)
    if item["type"] != "unanswerable":
        flags = useful_flags(item, [c["text"] for c in texts])
        src, ids = gold_sources(item), gold_chunks(item)
        gt_claims = extract_claims(q, item["ground_truth"])
        s.update(useful=flags, context_precision=float(np.mean(flags)) if flags else 0.0,
                 context_ap=average_precision(flags),
                 source_precision=float(np.mean([c["source"] in src for c in texts])) if texts else 0.0,
                 retrieval_hit=int(all(any(c["source"] == x for c in texts) for x in src)),
                 gold_chunk_recall=float(np.mean([any(c["chunk_id"] == x for c in texts) for x in ids])),
                 context_recall=verify(gt_claims, full) / len(gt_claims))
        if run.get("kg_context"):  # tỉ lệ quan hệ KG đưa vào context là hữu ích
            rel_lines = run["kg_context"].split("Knowledge graph relations:\n", 1)[-1].splitlines()
            rf = useful_flags(item, [x for x in rel_lines if x.strip()])
            s["kg_relation_precision"] = float(np.mean(rf)) if rf else math.nan
    else:
        s.update(context_precision=math.nan, context_ap=math.nan, source_precision=math.nan,
                 retrieval_hit=math.nan, gold_chunk_recall=math.nan, context_recall=math.nan)

    # Faithfulness / hallucination: trên context thực sự đưa vào LLM (chính) và chỉ trên đoạn văn (phụ)
    if refusal:
        s.update(n_claims=0, n_unsupported=0, faithfulness=math.nan, hallucination=0,
                 faithfulness_text=math.nan, hallucination_text=0)
    else:
        claims = extract_claims(q, ans)
        n, sup = len(claims), verify(claims, full)
        sup_text = verify(claims, texts) if len(full) > len(texts) else sup
        s.update(n_claims=n, n_unsupported=n - sup, faithfulness=sup / n, hallucination=int(sup < n),
                 faithfulness_text=sup_text / n, hallucination_text=int(sup_text < n), claims=claims)

    s["answer_relevancy"] = 0.0 if refusal else answer_relevancy(q, ans)
    s["answer_correctness"] = answer_correctness(item, ans, refusal)
    return s


MAIN = ["context_precision", "faithfulness", "answer_relevancy", "hallucination_rate", "answer_correctness"]
EXTRA = ["context_ap", "source_precision", "retrieval_hit", "gold_chunk_recall", "context_recall",
         "kg_relation_precision", "faithfulness_text", "hallucination_rate_text", "refusal_rate",
         "latency_s", "llm_calls"]
RENAME = {"hallucination": "hallucination_rate", "hallucination_text": "hallucination_rate_text",
          "refusal": "refusal_rate"}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (qtype, sys_name), g in df.groupby(["type", "system"], sort=False):
        row = {"type": qtype, "system": sys_name, "n": len(g)}
        for m in [*MAIN, *EXTRA]:
            col = {v: k for k, v in RENAME.items()}.get(m, m)
            if col not in g:
                continue
            mean, lo, hi = bootstrap_ci(g[col].tolist())
            row[m], row[f"{m}_ci95"] = round(mean, 4), f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def paired_diff(df: pd.DataFrame, a="graphrag", b="vector_rag", n=2000) -> pd.DataFrame:
    """Hiệu (a - b) theo từng câu hỏi + CI 95% bootstrap ghép cặp."""
    rows = []
    for qtype, g in df.groupby("type", sort=False):
        pa, pb = g[g.system == a].set_index("id"), g[g.system == b].set_index("id")
        ids = pa.index.intersection(pb.index)
        for m in MAIN:
            col = {v: k for k, v in RENAME.items()}.get(m, m)
            d = (pa.loc[ids, col] - pb.loc[ids, col]).dropna().to_numpy(dtype=float)
            if not len(d):
                continue
            rng = np.random.default_rng(config.SEED)
            means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            rows.append({"type": qtype, "metric": m, "n_pairs": len(d), "diff": round(d.mean(), 4),
                         "ci95": f"[{lo:.3f}, {hi:.3f}]", "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


NOTES = """\
- **context_precision** (retrieval precision, 0–1, cao = tốt): tỉ lệ đoạn văn (trong TOP_K đoạn mỗi hệ lấy ra)
  mà judge đánh giá là giúp suy ra đáp án chuẩn. Câu ngoài corpus không tính.
- **faithfulness / hallucination_rate**: như ý 1, chấm trên đúng context đưa vào LLM (GraphRAG: mô tả entity/
  quan hệ trong KG + đoạn văn). `*_text` (phụ): chỉ chấm trên đoạn văn.
- **answer_relevancy**: như ý 1 (câu từ chối = 0). **answer_correctness**: judge so với đáp án chuẩn
  (câu ngoài corpus: đúng = từ chối).
- **Hiệu ghép cặp**: graphrag − vector_rag trên cùng câu hỏi; `significant` = CI 95% không chứa 0.
"""


def write_summary_md(sm: pd.DataFrame, diff: pd.DataFrame, out: Path, info: dict):
    lines = [f"# Kết quả ý 2: GraphRAG vs Vector RAG — `{info['run']}`\n",
             f"{info['kg']['papers']} bài · KG: {info['kg']['entities']} entity, {info['kg']['relations']} quan hệ · "
             f"LLM: `{config.LLM_MODEL}` · Judge: `{config.JUDGE_MODEL}` · Embedding: `{config.EMBED_MODEL}` · "
             f"TOP_K = {config.TOP_K} đoạn văn · version: xem `run_info.json`\n",
             "## 1. Chỉ số chính\n"]
    types = [t for t in ("all", *TYPES) if t in set(sm["type"])]
    for t in types:
        g = sm[sm["type"] == t]
        cols = ["system", "n"] + [c for m in MAIN if m in g and g[m].notna().any() for c in (m, f"{m}_ci95")]
        lines += [f"### {TITLES[t]}\n", g[cols].to_markdown(index=False), ""]
    lines += ["*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*\n",
              "## 2. Hiệu ghép cặp (graphrag − vector_rag)\n", diff.to_markdown(index=False), "", NOTES,
              "## 3. Chỉ số phụ (chỉ để giải thích kết quả)\n"]
    for t in types:
        g = sm[sm["type"] == t]
        lines += [f"### {TITLES[t]}\n",
                  g[["system"] + [c for c in EXTRA if c in g and g[c].notna().any()]].to_markdown(index=False), ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="kg30", help="tên thư mục kết quả trong results/")
    ap.add_argument("--testset", nargs="+", default=[str(f) for f in TESTSET_FILES])
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    papers = load_papers(config.KG_K)
    use_kg_store()  # Vector RAG truy xuất trên Chroma riêng của ý 2 (đúng 30 bài); đặt trước khi dựng hệ thống
    scope = set(papers)
    testset = [t for t in load_testset([f for f in args.testset if Path(f).exists()], args.include_unreviewed)
               if t["type"] == "unanswerable" or set(gold_sources(t)) <= scope]
    if args.limit:
        testset = testset[: args.limit]
    if not testset:
        raise SystemExit("Không có câu hỏi nào thuộc 30 bài.")
    out = config.RESULTS_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)

    from rag_graphrag import get_kg
    kg_stats = json.loads((config.KG_DIR / f"k{config.KG_K}" / "graph.json").read_text(encoding="utf-8"))["stats"]
    info = {"run": args.run, "started": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
            "top_k": config.TOP_K, "num_ctx": config.NUM_CTX, "kg": kg_stats,
            "graphrag": {"seed_entities": config.KG_SEED_ENTITIES, "max_relations": config.KG_MAX_RELATIONS},
            "testset": {Path(fp).name: hashlib.sha256(Path(fp).read_bytes()).hexdigest()[:16]
                        for fp in args.testset if Path(fp).exists()},
            "n_by_type": pd.Series([t["type"] for t in testset]).value_counts().to_dict(),
            "environment": env_info()}
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(testset)} câu hỏi {info['n_by_type']} | {len(papers)} bài | top-k={config.TOP_K} -> {out}")
    get_kg()  # nạp KG trước để báo lỗi sớm nếu chưa dựng

    rows = []
    for sys_name in args.systems:
        module = __import__(SYSTEMS[sys_name])
        ans_path, sc_path = out / f"answers_{sys_name}.jsonl", out / f"scores_{sys_name}.jsonl"
        answers, scores = load_jsonl(ans_path), load_jsonl(sc_path)
        for item in tqdm(testset, desc=f"[{sys_name}] trả lời"):
            if item["id"] not in answers:
                answers[item["id"]] = {"id": item["id"], **module.answer(item["question"])}
                append_jsonl(ans_path, answers[item["id"]])
        for item in tqdm(testset, desc=f"[{sys_name}] chấm điểm"):
            if item["id"] not in scores:
                scores[item["id"]] = {"id": item["id"], **score(item, answers[item["id"]])}
                append_jsonl(sc_path, scores[item["id"]])
        for item in testset:
            a, s = answers[item["id"]], scores[item["id"]]
            rows.append({"system": sys_name, "id": item["id"], "type": item["type"], "question": item["question"],
                         "answer": a["answer"], "latency_s": a["latency_s"], "llm_calls": a["llm_calls"],
                         **{k: v for k, v in s.items() if k not in ("id", "claims", "useful")}})

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_question.csv", index=False, encoding="utf-8-sig")
    answerable = df[df["type"] != "unanswerable"]
    full = pd.concat([answerable.assign(type="all"), df]) if answerable["type"].nunique() > 1 else df
    sm, diff = summarize(full), (paired_diff(full) if len(args.systems) == 2 else pd.DataFrame())
    sm.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(out / "paired_diff.csv", index=False, encoding="utf-8-sig")
    write_summary_md(sm, diff, out / "summary.md", info)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
