"""Bước 3: chạy 2 hệ thống trên cùng bộ câu hỏi và chấm điểm bằng LLM-judge local.

    python evaluate.py --run k4                  # top-k = 4, cả 3 loại câu hỏi
    python evaluate.py --run k8 --top-k 8 --testset testset/testset.json testset/multihop.json
                                                 # retrieval nhiễu: top-k lớn, nhiều đoạn không liên quan
    python evaluate.py --run thu --include-unreviewed --limit 5   # chạy thử

Mỗi lần chạy lưu vào results/<run>/ (answers_*, scores_*, per_question.csv, summary.*, run_info.json
gồm version thư viện / Ollama / model). Có thể dừng giữa chừng (Ctrl+C) và chạy lại: kết quả đã có được giữ.

Loại câu hỏi (trường "type" trong testset):
  - single        : hỏi sự kiện trong 1 đoạn văn (testset/testset.json)
  - multihop      : cần tổng hợp thông tin từ 2 bài báo khác nhau (testset/multihop.json, trường "sources")
  - unanswerable  : câu hỏi ngoài corpus, hành vi đúng là từ chối trả lời (testset/unanswerable.json)

Ba chỉ số chính của đề tài (định nghĩa theo RAGAS, Es et al. 2023):
  - faithfulness       = #claim được context hỗ trợ / #claim trong câu trả lời
  - answer_relevancy   = trung bình cosine(câu hỏi gốc, 3 câu hỏi sinh ngược từ câu trả lời);
                         = 0 nếu câu trả lời né tránh (noncommittal / từ chối)
  - hallucination_rate = tỉ lệ câu hỏi có câu trả lời chứa >= 1 claim KHÔNG được context hỗ trợ
Context để chấm faithfulness / hallucination: full top-k ban đầu của câu hỏi gốc, giống hệt nhau cho cả 2 hệ
(Self-RAG lọc bớt tài liệu trước khi sinh; nếu chấm trên tài liệu đã lọc thì Self-RAG bị chấm trên ít bằng
chứng hơn -> không công bằng). Cột *_used (phụ) chấm trên tài liệu thực sự đưa vào generate.
Chỉ số phụ (chỉ để giải thích): refusal_rate (câu ngoài corpus: tỉ lệ abstain đúng), answer_correctness
(so với đáp án chuẩn; câu ngoài corpus: đúng = từ chối), retrieval_hit, claim_halluc_rate,
halluc_rate_among_answered, latency_s, llm_calls.
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ollama import ResponseError
from tqdm import tqdm

import config
from common import docs_to_contexts, env_info, get_embeddings, get_llm, get_retriever, is_refusal

SYSTEMS = {
    "langchain_rag": "rag_langchain",
    "langgraph_selfrag": "rag_selfrag",
}
TYPES = ("single", "multihop", "unanswerable")

# ---------------- Judge prompts ----------------
EXTRACT_CLAIMS = ChatPromptTemplate.from_messages([
    ("system", "Break the answer into a list of short, atomic, self-contained factual statements "
               "(claims). Resolve pronouns. Ignore filler phrases. "
               'Return JSON: {{"claims": ["...", "..."]}}'),
    ("human", "Question: {question}\nAnswer: {answer}"),
])

VERIFY_CLAIMS = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact checker. For each numbered statement, decide whether it can be "
               "directly inferred from the context. Use 'yes' only if the context supports it; "
               "use 'no' if it is contradicted OR not mentioned. "
               'Return JSON: {{"verdicts": [{{"id": 1, "supported": "yes" or "no"}}, ...]}} '
               "with one entry per statement."),
    ("human", "Context:\n{context}\n\nStatements:\n{statements}"),
])

REVERSE_QUESTIONS = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 different questions that the given answer would answer. "
               "Also say whether the answer is noncommittal (evasive, vague, 'I don't know', "
               "'cannot find'). "
               'Return JSON: {{"questions": ["...", "...", "..."], "noncommittal": true or false}}'),
    ("human", "Answer: {answer}"),
])

CORRECTNESS = ChatPromptTemplate.from_messages([
    ("system", "Compare a candidate answer with a reference answer to the question. "
               "Score 1 if the candidate conveys the key facts of the reference without contradicting "
               "it, otherwise 0. "
               'Return JSON: {{"score": 0 or 1}}'),
    ("human", "Question: {question}\nReference: {reference}\nCandidate: {answer}"),
])


def judge(prompt, **kw) -> dict:
    try:
        raw = (prompt | get_llm(config.JUDGE_MODEL, json_mode=True) | StrOutputParser()).invoke(kw)
        return json.loads(raw)
    except (json.JSONDecodeError, ResponseError):  # JSON hỏng / Ollama huỷ do lặp token
        return {}


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def initial_contexts(question: str) -> list[dict]:
    """Full top-k ban đầu cho câu hỏi gốc: context chấm điểm chung cho mọi hệ."""
    return docs_to_contexts(get_retriever().invoke(question))


def verify(claims: list[str], contexts: list[dict]) -> int:
    """Số claim được context hỗ trợ (thiếu verdict = không hỗ trợ)."""
    ctx = "\n\n".join(c["text"] for c in contexts) or "(no context)"
    stmts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    verdicts = judge(VERIFY_CLAIMS, context=ctx, statements=stmts).get("verdicts", [])
    sup = {int(v.get("id", 0)): str(v.get("supported", "")).lower().startswith("y")
           for v in verdicts if isinstance(v, dict) and str(v.get("id", "")).isdigit()}
    return sum(sup.get(i + 1, False) for i in range(len(claims)))


def score(item: dict, run: dict) -> dict:
    q, ans = item["question"], run["answer"]
    qtype = item.get("type", "single")
    sources = [] if qtype == "unanswerable" else (item.get("sources") or [item["source"]])
    used, initial = run["contexts"], initial_contexts(q)
    refusal = is_refusal(ans)
    s = {"refusal": int(refusal),
         "retrieval_hit": (int(all(any(c["source"] == src for c in used) for src in sources))
                           if sources else math.nan)}

    # Faithfulness + hallucination: trên top-k ban đầu (chính) và trên tài liệu đưa vào generate (phụ)
    if refusal:
        s.update(n_claims=0, n_unsupported=0, faithfulness=math.nan, hallucination=0,
                 n_unsupported_used=0, faithfulness_used=math.nan, hallucination_used=0)
    else:
        claims = [c for c in judge(EXTRACT_CLAIMS, question=q, answer=ans).get("claims", [])
                  if isinstance(c, str) and c.strip()] or [ans]
        sup = verify(claims, initial)
        same = [c["text"] for c in used] == [c["text"] for c in initial]
        sup_used = sup if same else verify(claims, used)
        n = len(claims)
        s.update(n_claims=n, n_unsupported=n - sup, faithfulness=sup / n, hallucination=int(sup < n),
                 n_unsupported_used=n - sup_used, faithfulness_used=sup_used / n,
                 hallucination_used=int(sup_used < n), claims=claims)

    # Answer relevancy
    if refusal:
        s["answer_relevancy"] = 0.0
    else:
        r = judge(REVERSE_QUESTIONS, answer=ans)
        gen_qs = [x for x in r.get("questions", []) if isinstance(x, str) and x.strip()][:3]
        if r.get("noncommittal") is True or not gen_qs:
            s["answer_relevancy"] = 0.0
        else:
            emb = get_embeddings()
            qv, gvs = emb.embed_query(q), emb.embed_documents(gen_qs)
            s["answer_relevancy"] = float(np.mean([cosine(qv, g) for g in gvs]))

    # Correctness: so với đáp án chuẩn; câu ngoài corpus thì đúng = từ chối
    if qtype == "unanswerable":
        s["answer_correctness"] = int(refusal)
    else:
        s["answer_correctness"] = 0 if refusal else int(
            judge(CORRECTNESS, question=q, reference=item["ground_truth"], answer=ans).get("score", 0) in (1, "1"))
    return s


NOTES = """\
- **faithfulness** (0–1, cao = tốt): câu trả lời được tách thành các claim nguyên tử; judge kiểm tra từng
  claim có suy ra được từ context không. Điểm = #claim được hỗ trợ / #claim. Câu từ chối không tính.
- **answer_relevancy** (0–1, cao = tốt): judge sinh ngược 3 câu hỏi từ câu trả lời; điểm = trung bình
  cosine (embedding) với câu hỏi gốc. Câu né tránh/từ chối = 0. Không dùng cho câu ngoài corpus.
- **hallucination_rate** (0–1, thấp = tốt): tỉ lệ câu hỏi có câu trả lời chứa ≥ 1 claim không được
  context hỗ trợ, tính trên toàn bộ câu hỏi.
- **Context chấm điểm**: full top-k ban đầu của câu hỏi gốc, giống nhau cho cả 2 hệ. Cột `*_used` chấm
  trên tài liệu thực sự đưa vào generate (Self-RAG: tài liệu đã lọc).
- **refusal_rate ở câu ngoài corpus** (cao = tốt): tỉ lệ hệ thống từ chối trả lời đúng (abstain).
"""


# ---------------- IO helpers ----------------
def load_jsonl(path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f) if d}


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_testset(files, include_unreviewed=False) -> list[dict]:
    items = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for t in json.load(f):
                if t.get("keep", True) and (include_unreviewed or t.get("reviewed")):
                    items.append({"type": "single", **t})
    ids = [t["id"] for t in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("Trùng id câu hỏi giữa các file testset")
    return items


def bootstrap_ci(x, n=2000, seed=config.SEED):
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["faithfulness", "answer_relevancy", "hallucination", "faithfulness_used",
               "hallucination_used", "answer_correctness", "refusal", "retrieval_hit", "latency_s", "llm_calls"]
    rename = {"hallucination": "hallucination_rate", "hallucination_used": "hallucination_rate_used",
              "refusal": "refusal_rate"}
    rows = []
    for (qtype, sys_name), g in df.groupby(["type", "system"], sort=False):
        row = {"type": qtype, "system": sys_name, "n": len(g)}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].tolist())
            name = rename.get(m, m)
            row[name] = round(mean, 4)
            row[f"{name}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        row["claim_halluc_rate"] = round(g["n_unsupported"].sum() / max(g["n_claims"].sum(), 1), 4)
        answered = g[g["refusal"] == 0]
        row["halluc_rate_among_answered"] = round(answered["hallucination"].mean(), 4) if len(answered) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


MAIN = ["faithfulness", "answer_relevancy", "hallucination_rate"]
TITLES = {"all": "Tất cả câu có đáp án (single-hop + multi-hop)", "single": "Câu hỏi một đoạn (single-hop)",
          "multihop": "Câu hỏi tổng hợp 2 bài (multi-hop)",
          "unanswerable": "Câu hỏi ngoài corpus (đúng = từ chối; answer_relevancy của câu từ chối = 0)"}


def write_summary_md(sm: pd.DataFrame, out: Path, info: dict):
    lines = [f"# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG — `{info['run']}`\n",
             f"LLM: `{config.LLM_MODEL}` · Judge: `{config.JUDGE_MODEL}` · Embedding: `{config.EMBED_MODEL}` · "
             f"top-k={config.TOP_K} · version: xem `run_info.json`\n",
             "## 1. Ba chỉ số chính\n"]
    extra = ["system", "faithfulness_used", "hallucination_rate_used", "claim_halluc_rate",
             "halluc_rate_among_answered", "answer_correctness", "refusal_rate", "retrieval_hit",
             "latency_s", "llm_calls"]
    types = [t for t in ("all", *TYPES) if t in set(sm["type"])]
    for t in types:
        g = sm[sm["type"] == t]
        lines += [f"### {TITLES[t]}\n",
                  g[["system", "n"] + [c for m in MAIN for c in (m, f"{m}_ci95")]].to_markdown(index=False), ""]
    lines += ["*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*\n", NOTES,
              "## 2. Chỉ số phụ (chỉ để giải thích kết quả)\n"]
    for t in types:
        g = sm[sm["type"] == t]
        lines += [f"### {TITLES[t]}\n", g[[c for c in extra if c in g]].to_markdown(index=False), ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="k4", help="tên thư mục kết quả trong results/")
    ap.add_argument("--testset", nargs="+", default=[str(f) for f in config.TESTSET_FILES])
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--top-k", type=int, default=0, help="ghi đè TOP_K cho cả 2 hệ (vd. 8 = retrieval nhiễu)")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.top_k:
        config.TOP_K = args.top_k  # đặt trước khi dựng 2 hệ thống

    testset = load_testset(args.testset, args.include_unreviewed)
    if args.limit:
        testset = testset[: args.limit]
    if not testset:
        raise SystemExit("Không có câu hỏi nào. Duyệt testset (reviewed=true) hoặc thêm --include-unreviewed.")
    out = config.RESULTS_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)
    info = {"run": args.run, "started": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
            "top_k": config.TOP_K, "num_ctx": config.NUM_CTX,
            "self_rag": {"max_query_rewrites": config.MAX_QUERY_REWRITES,
                         "max_regenerations": config.MAX_REGENERATIONS},
            "testset": {Path(fp).name: hashlib.sha256(Path(fp).read_bytes()).hexdigest()[:16] for fp in args.testset},
            "n_by_type": pd.Series([t["type"] for t in testset]).value_counts().to_dict(),
            "environment": env_info()}
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(testset)} câu hỏi {info['n_by_type']} | LLM={config.LLM_MODEL} | judge={config.JUDGE_MODEL} "
          f"| top-k={config.TOP_K} -> {out}")

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
            rows.append({"system": sys_name, "id": item["id"], "type": item["type"], "topic": item.get("topic"),
                         "question": item["question"], "answer": a["answer"],
                         "latency_s": a["latency_s"], "llm_calls": a["llm_calls"],
                         **{k: v for k, v in s.items() if k not in ("id", "claims")}})

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_question.csv", index=False, encoding="utf-8-sig")
    # Bảng gộp chỉ gồm câu có đáp án: câu ngoài corpus từ chối đúng nhưng answer_relevancy = 0 -> gộp vào sẽ lệch
    answerable = df[df["type"] != "unanswerable"]
    sm = summarize(pd.concat([answerable.assign(type="all"), df])
                   if answerable["type"].nunique() > 1 else df)
    sm.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    write_summary_md(sm, out / "summary.md", info)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
