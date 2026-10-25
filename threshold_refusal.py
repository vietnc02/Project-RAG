"""Baseline "RAG thuần + refuse theo ngưỡng cosine" (bản đánh giá, Y1.3): Self-RAG có tốt hơn một quy tắc refuse rẻ
tiền ở cùng coverage không?

    python threshold_refusal.py results         # đọc <root>/k4, k8 -> <root>/threshold_refusal.json + .csv

Với mỗi câu hỏi, điểm truy xuất s = cosine(câu hỏi, đoạn đứng đầu top-k) (cùng Chroma, cùng embedding của ý 1). Quy
tắc: refuse nếu s < τ, còn lại giữ nguyên câu trả lời của LangChain RAG thuần (đã chấm sẵn), nên không cần gọi LLM.
Quét τ cho cả đường risk–coverage của RAG thuần:
  coverage = tỉ lệ câu có đáp án được trả lời; risk = tỉ lệ câu trả lời sai hoặc có claim không được hỗ trợ.
So với Self-RAG tại cùng coverage: chọn τ sao cho coverage của RAG thuần + ngưỡng bằng coverage của Self-RAG (riêng
từng loại câu hỏi), rồi so hallucination và độ đúng trên câu được trả lời, số câu đúng và có căn cứ.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from common import get_vectorstore
from evaluate import load_jsonl, load_testset


def top1_similarity(questions: dict[str, str], k: int) -> dict[str, float]:
    vs = get_vectorstore()
    out = {}
    for i, q in questions.items():
        docs = vs.similarity_search_with_score(q, k=k)   # khoảng cách cosine của Chroma
        out[i] = 1 - min(d for _, d in docs)
    return out


def label(s: dict) -> str:
    if s["refusal"]:
        return "refused"
    if s["answer_correctness"]:
        return "correct_unsupported" if s["hallucination"] else "grounded_correct"
    return "wrong"


def curve(df: pd.DataFrame) -> pd.DataFrame:
    """Quét τ trên các giá trị s có thật: mỗi τ -> coverage, risk, hallucination / correctness trên câu trả lời."""
    rows = []
    for tau in [-np.inf, *sorted(df.sim.unique()), np.inf]:
        ans = df[(df.sim >= tau) & (df.label != "refused")]
        n_ans = len(ans)
        rows.append({"tau": tau, "coverage": n_ans / len(df), "n_answered": n_ans,
                     "risk": ((ans.label == "correct_unsupported") | (ans.label == "wrong")).mean() if n_ans else math.nan,
                     "halluc_answered": ans.hallucination.mean() if n_ans else math.nan,
                     "correct_answered": ans.answer_correctness.mean() if n_ans else math.nan,
                     "grounded_correct": int((ans.label == "grounded_correct").sum())})
    return pd.DataFrame(rows)


def main(root: Path):
    items = {t["id"]: t for t in load_testset(config.TESTSET_FILES) if t["type"] != "unanswerable"}
    res, curves = {}, []
    for run in ("k4", "k8"):
        d = root / run
        lc, sr = load_jsonl(d / "scores_langchain_rag.jsonl"), load_jsonl(d / "scores_langgraph_selfrag.jsonl")
        if not lc or not sr:
            continue
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        ids = [i for i in items if i in lc and i in sr]
        sims = top1_similarity({i: items[i]["question"] for i in ids}, info["top_k"])
        df = pd.DataFrame([{"id": i, "type": items[i]["type"], "sim": sims[i], "label": label(lc[i]),
                            "hallucination": lc[i]["hallucination"], "answer_correctness": lc[i]["answer_correctness"]}
                           for i in ids])
        res[run] = {}
        for qt, g in df.groupby("type"):
            c = curve(g).assign(run=run, type=qt)
            curves.append(c)
            srg = [sr[i] for i in g.id]
            sr_ans = [s for s in srg if not s["refusal"]]
            sr_pt = {"coverage": len(sr_ans) / len(srg),
                     "risk": float(np.mean([label(s) != "grounded_correct" for s in sr_ans])),
                     "halluc_answered": float(np.mean([s["hallucination"] for s in sr_ans])),
                     "correct_answered": float(np.mean([s["answer_correctness"] for s in sr_ans])),
                     "grounded_correct": sum(label(s) == "grounded_correct" for s in sr_ans), "n_answered": len(sr_ans)}
            # τ cho RAG thuần + ngưỡng: coverage gần coverage của Self-RAG nhất (ưu tiên không cao hơn)
            m = c.iloc[(c.coverage - sr_pt["coverage"]).abs().argsort().iloc[0]]
            res[run][qt] = {"n": len(g), "self_rag": sr_pt,
                            "rag_threshold": {k: (float(m[k]) if k != "grounded_correct" else int(m[k]))
                                              for k in ("tau", "coverage", "risk", "halluc_answered",
                                                        "correct_answered", "grounded_correct", "n_answered")},
                            "rag_no_threshold": {k: float(c.iloc[0][k]) for k in ("coverage", "risk", "halluc_answered",
                                                                                   "correct_answered")}}
    (root / "threshold_refusal.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                                 encoding="utf-8")
    pd.concat(curves).to_csv(root / "threshold_refusal_curve.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else config.RESULTS_DIR))
