"""Ý 3: KG dựng từ k = 10 / 20 / 30 bài — k bài có đủ để KG (GraphRAG) hoạt động tốt không?

    python kg_build.py --build --k 10 && python kg_build.py --build --k 20   # KG k10, k20 (k30 có sẵn từ ý 2)
    python kg_quality.py --all                    # chất lượng KG (entity coverage, relation completeness)
    python evaluate_kg_scale.py                   # chất lượng câu trả lời của GraphRAG với KG k = 10, 20, 30
    python evaluate_kg_scale.py --report-only     # chỉ tổng hợp lại bảng từ kết quả đã có

KG k bài = KG dựng từ k bài ĐẦU trong data/kg/papers.json (thứ tự phân tầng theo chủ đề) bằng đúng cache trích
xuất của ý 2 -> KG10 ⊂ KG20 ⊂ KG30 lồng nhau, khác biệt giữa các k chỉ đến từ số bài báo. Hệ thống trả lời là
GraphRAG của ý 2 (rag_graphrag.py, cùng prompt / LLM / TOP_K), chỉ đổi KG.

Bộ câu hỏi: đúng 85 câu của ý 2 (62 câu có đáp án thuộc 30 bài + 23 câu ngoài corpus). Mỗi câu có đáp án được gán
phạm vi (scope) = k nhỏ nhất trong {10, 20, 30} mà mọi bài nguồn của câu nằm trong k bài đầu:
  - "trong phạm vi KG" (scope <= k): câu hỏi về các bài có trong KG  (n = 15 / 37 / 62 với k = 10 / 20 / 30)
  - tập cố định S10 (scope = 10, 15 câu) và S20 (scope <= 20, 37 câu): cùng câu hỏi, KG lớn dần -> so sánh
    ghép cặp, tách riêng tác động của việc thêm bài vào KG
  - toàn bộ 62 câu: KG k bài trả lời được bao nhiêu câu hỏi về cả chủ đề (câu ngoài phạm vi KG -> mong đợi từ chối)
Chỉ số giống ý 2 (evaluate_graphrag.score): context_precision, faithfulness, answer_relevancy, hallucination_rate,
answer_correctness. Kết quả GraphRAG k = 30 lấy lại từ results/kg30 của ý 2 (cùng code, cùng KG, cùng câu hỏi).
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from common import env_info
# Dùng lại nguyên bộ câu hỏi, hàm chấm và hàm tổng hợp của ý 2 (evaluate_graphrag.py không bị sửa)
from evaluate import append_jsonl, bootstrap_ci, load_jsonl, load_testset
from evaluate_graphrag import EXTRA, MAIN, RENAME, TESTSET_FILES, gold_sources, paired_diff, score, summarize
from kg_build import load_papers

KS = (10, 20, 30)
OUT = config.RESULTS_DIR / "kg_scale"
K30_FROM_Y2 = config.RESULTS_DIR / "kg30"
SYSTEM = "graphrag"


def load_items() -> list[dict]:
    papers = load_papers(max(KS))
    rank = {s: i + 1 for i, s in enumerate(papers)}
    items = [t for t in load_testset([f for f in TESTSET_FILES if f.exists()])
             if t["type"] == "unanswerable" or set(gold_sources(t)) <= set(papers)]
    for t in items:
        t["scope"] = 0 if t["type"] == "unanswerable" else next(
            k for k in KS if max(rank[s] for s in gold_sources(t)) <= k)
    return items


def run_k(k: int, items: list[dict]):
    out = OUT / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    ans_path, sc_path = out / f"answers_{SYSTEM}.jsonl", out / f"scores_{SYSTEM}.jsonl"
    if k == 30 and not ans_path.exists() and (K30_FROM_Y2 / f"answers_{SYSTEM}.jsonl").exists():
        for p in (ans_path, sc_path):
            shutil.copy(K30_FROM_Y2 / p.name, p)
        print(f"k=30: lấy kết quả GraphRAG của ý 2 từ {K30_FROM_Y2}")

    import rag_graphrag
    config.KG_K = k
    rag_graphrag.get_kg.cache_clear()
    rag_graphrag.get_kg()  # nạp KG trước để báo lỗi sớm nếu chưa dựng
    kg_stats = json.loads((config.KG_DIR / f"k{k}" / "graph.json").read_text(encoding="utf-8"))["stats"]
    info = {"k": k, "started": datetime.now().isoformat(timespec="seconds"), "top_k": config.TOP_K,
            "num_ctx": config.NUM_CTX, "kg": kg_stats,
            "graphrag": {"seed_entities": config.KG_SEED_ENTITIES, "max_relations": config.KG_MAX_RELATIONS},
            "testset": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in TESTSET_FILES if f.exists()},
            "environment": env_info()}
    if k == 30 and (K30_FROM_Y2 / "run_info.json").exists():
        info["answers_from"] = str(K30_FROM_Y2.relative_to(config.PROJECT_DIR))
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    answers, scores = load_jsonl(ans_path), load_jsonl(sc_path)
    for item in tqdm(items, desc=f"[KG k={k}] trả lời"):
        if item["id"] not in answers:
            answers[item["id"]] = {"id": item["id"], **rag_graphrag.answer(item["question"])}
            append_jsonl(ans_path, answers[item["id"]])
    for item in tqdm(items, desc=f"[KG k={k}] chấm điểm"):
        if item["id"] not in scores:
            scores[item["id"]] = {"id": item["id"], **score(item, answers[item["id"]])}
            append_jsonl(sc_path, scores[item["id"]])


def collect(items: list[dict]) -> pd.DataFrame:
    rows = []
    for k in KS:
        out = OUT / f"k{k}"
        answers, scores = load_jsonl(out / f"answers_{SYSTEM}.jsonl"), load_jsonl(out / f"scores_{SYSTEM}.jsonl")
        for item in items:
            if item["id"] not in scores:
                continue
            a, s = answers[item["id"]], scores[item["id"]]
            rows.append({"k": k, "system": f"k{k}", "id": item["id"], "type": item["type"], "scope": item["scope"],
                         "in_scope": int(0 < item["scope"] <= k), "question": item["question"],
                         "answer": a["answer"], "kg_entities": len(a.get("kg_entities", [])),
                         "latency_s": a["latency_s"], "llm_calls": a["llm_calls"],
                         **{c: v for c, v in s.items() if c not in ("id", "claims", "useful")}})
    return pd.DataFrame(rows)


SET_TITLES = {
    "in_scope": "Câu hỏi trong phạm vi KG (bài nguồn nằm trong k bài)",
    "in_scope_single": "Trong phạm vi KG — single-hop",
    "in_scope_multihop": "Trong phạm vi KG — multi-hop",
    "S10": "Tập cố định S10 (15 câu về 10 bài đầu)",
    "S20": "Tập cố định S20 (37 câu về 20 bài đầu)",
    "all": "Toàn bộ 62 câu có đáp án (cả câu ngoài phạm vi KG)",
    "out_scope": "Câu ngoài phạm vi KG (bài nguồn chưa có trong KG)",
    "unanswerable": "Câu hỏi ngoài corpus (đúng = từ chối)",
}


def question_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn nhãn tập câu hỏi (cột type) để dùng lại summarize / paired_diff của ý 2 (nhóm theo type, system)."""
    ans = df[df["type"] != "unanswerable"]
    parts = [ans[ans.in_scope == 1].assign(type="in_scope"),
             ans[(ans.in_scope == 1) & (ans.type == "single")].assign(type="in_scope_single"),
             ans[(ans.in_scope == 1) & (ans.type == "multihop")].assign(type="in_scope_multihop"),
             ans[ans.scope == 10].assign(type="S10"), ans[ans.scope <= 20].assign(type="S20"),
             ans.assign(type="all"), ans[ans.in_scope == 0].assign(type="out_scope"),
             df[df["type"] == "unanswerable"]]
    return pd.concat(parts, ignore_index=True)


def fixed_set_diffs(sets: pd.DataFrame) -> pd.DataFrame:
    """Hiệu ghép cặp trên cùng câu hỏi: KG nhỏ − KG30 (và KG10 − KG20 trên S10)."""
    out = []
    for name, a, b in (("S10", "k10", "k30"), ("S10", "k20", "k30"), ("S10", "k10", "k20"), ("S20", "k20", "k30"),
                       ("all", "k10", "k30"), ("all", "k20", "k30")):
        g = sets[(sets.type == name) & sets.system.isin([a, b])]
        d = paired_diff(g, a=a, b=b)
        if len(d):
            out.append(d.assign(compare=f"{a} − {b}"))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return df[["type", "compare", "metric", "n_pairs", "diff", "ci95", "significant"]] if len(df) else df


def coverage_vs_answer(df: pd.DataFrame) -> pd.DataFrame | None:
    """Câu trả lời đúng hay sai theo mức độ KG phủ thực thể của câu hỏi (cần kg_quality.py --coverage)."""
    fp = OUT / "kg_coverage_per_question.csv"
    if not fp.exists():
        return None
    cov = pd.read_csv(fp)
    m = df[df.type != "unanswerable"].merge(cov[["k", "id", "entity_coverage"]], on=["k", "id"])
    m["covered"] = np.where(m.entity_coverage >= 1, "đủ thực thể", np.where(m.entity_coverage > 0, "thiếu một phần",
                                                                           "không có thực thể nào"))
    rows = []
    for (k, c), g in m.groupby(["k", "covered"]):
        mean, lo, hi = bootstrap_ci(g["answer_correctness"].tolist())
        rows.append({"k": k, "entity_coverage_of_question": c, "n": len(g), "answer_correctness": round(mean, 4),
                     "answer_correctness_ci95": f"[{lo:.3f}, {hi:.3f}]",
                     "refusal_rate": round(g["refusal"].mean(), 4)})
    return pd.DataFrame(rows)


NOTES = """\
- **KG k bài**: dựng từ k bài đầu của `data/kg/papers.json` bằng cùng cache trích xuất (KG10 ⊂ KG20 ⊂ KG30).
- **Trong phạm vi KG**: câu hỏi mà mọi bài nguồn đều nằm trong k bài -> mỗi k là một tập câu khác nhau.
  **S10 / S20**: tập câu cố định, cho phép so sánh ghép cặp giữa các k (`significant` = CI 95% không chứa 0).
- Chỉ số giống ý 2: context_precision (retrieval precision), faithfulness, answer_relevancy, hallucination_rate,
  answer_correctness. Câu ngoài phạm vi KG: hành vi tốt nhất là từ chối (không có tài liệu nguồn).
"""


def write_report(items: list[dict]):
    df = collect(items)
    if df.empty:
        raise SystemExit("Chưa có kết quả: chạy python evaluate_kg_scale.py")
    df.to_csv(OUT / "per_question.csv", index=False, encoding="utf-8-sig")
    sets = question_sets(df)
    sm = summarize(sets)
    sm.insert(2, "k", sm.system.str[1:].astype(int))
    diff = fixed_set_diffs(sets)
    cva = coverage_vs_answer(df)
    sm.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(OUT / "paired_diff.csv", index=False, encoding="utf-8-sig")
    if cva is not None:
        cva.to_csv(OUT / "coverage_vs_answer.csv", index=False, encoding="utf-8-sig")

    lines = ["# Kết quả ý 3: KG từ k = 10 / 20 / 30 bài\n",
             f"GraphRAG (ý 2) với KG dựng từ k bài đầu · LLM: `{config.LLM_MODEL}` · Judge: `{config.JUDGE_MODEL}` · "
             f"TOP_K = {config.TOP_K} · version: xem `k*/run_info.json`\n"]
    kq = OUT / "kg_quality.md"
    if kq.exists():
        lines += ["## 1. Chất lượng KG (chi tiết: `kg_quality.md`)\n",
                  kq.read_text(encoding="utf-8").split("<!-- main -->", 2)[1].strip(), ""]
    lines.append("## 2. Chất lượng câu trả lời\n")
    for t in SET_TITLES:
        g = sm[sm["type"] == t]
        if g.empty:
            continue
        cols = ["k", "n"] + [c for m in [*MAIN, "refusal_rate"] if m in g and g[m].notna().any()
                             for c in (m, f"{m}_ci95")]
        lines += [f"### {SET_TITLES[t]}\n", g[cols].to_markdown(index=False), ""]
    lines += ["*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*\n",
              "## 3. Hiệu ghép cặp trên cùng câu hỏi\n", diff.to_markdown(index=False), "", NOTES]
    if cva is not None:
        lines += ["## 4. Độ phủ thực thể của câu hỏi và độ đúng của câu trả lời\n", cva.to_markdown(index=False), ""]
    lines.append("## 5. Chỉ số phụ (chỉ để giải thích kết quả)\n")
    for t in ("in_scope", "S10", "all"):
        g = sm[sm["type"] == t]
        lines += [f"### {SET_TITLES[t]}\n",
                  g[["k"] + [c for c in EXTRA if c in g and g[c].notna().any()]].to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=list(KS), choices=list(KS))
    ap.add_argument("--limit", type=int, default=0, help="chạy thử trên n câu đầu")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} câu hỏi | phạm vi: "
          f"{pd.Series([t['scope'] for t in items]).value_counts().sort_index().to_dict()} (0 = ngoài corpus)")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        for k in args.k:
            run_k(k, items)
    write_report(items)


if __name__ == "__main__":
    main()
