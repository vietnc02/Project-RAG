"""Chỉ số truy xuất không cần judge, theo thứ hạng (bản đánh giá: Y1.1, Y2.1).

    python retrieval_metrics.py results/r2     # -> results/r2/retrieval_metrics.json, retrieval_metrics.csv

Gold chunk = đoạn dùng để sinh câu hỏi (single-hop: 1 đoạn; multi-hop: 2 đoạn của 2 bài). Với danh sách đoạn hệ thống
đưa vào LLM theo thứ tự xếp hạng:
  - hit@k     : có ít nhất một gold chunk trong k đoạn
  - recall@k  : tỉ lệ gold chunk lấy được (multi-hop: cần cả hai đoạn mới bằng 1)
  - MRR       : 1 / thứ hạng của gold chunk đầu tiên (0 nếu không lấy được)
  - nDCG@k    : độ liên quan nhị phân (gold = 1), chuẩn hoá theo xếp hạng lý tưởng min(#gold, k)
  - source_hit: có đủ mọi bài nguồn (như retrieval_hit)
Gold chunk chỉ là một đoạn nên các chỉ số này là cận dưới: đoạn khác cũng có thể chứa đáp án.
Self-RAG (ý 1) tính trên full top-k ban đầu (retrieved_contexts), tức đúng bước truy xuất chung của hai hệ.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from evaluate import load_jsonl, load_testset
from evaluate_graphrag import TESTSET_FILES as Y2_FILES, gold_chunks, gold_sources

RUNS = {  # thư mục -> {hệ: (file câu trả lời, trường chứa danh sách đoạn)}
    "k4": {"langchain_rag": ("answers_langchain_rag.jsonl", "contexts"),
           "langgraph_selfrag": ("answers_langgraph_selfrag.jsonl", "retrieved_contexts")},
    "kg30": {"vector_rag": ("answers_vector_rag.jsonl", "contexts"), "graphrag": ("answers_graphrag.jsonl", "contexts")},
}
RUNS["k8"] = RUNS["noise4"] = RUNS["k4"]
for k in (10, 20, 30):
    RUNS[f"kg_scale/k{k}"] = RUNS["kg30"]


def metrics(ctx: list[dict], item: dict) -> dict:
    gold, src = set(gold_chunks(item)), gold_sources(item)
    ranks = [i + 1 for i, c in enumerate(ctx) if c.get("chunk_id") in gold]
    k = len(ctx)
    dcg = sum(1 / math.log2(r + 1) for r in ranks)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return {"k": k, "hit": int(bool(ranks)), "recall": len({ctx[r - 1]["chunk_id"] for r in ranks}) / len(gold),
            "mrr": 1 / ranks[0] if ranks else 0.0, "ndcg": dcg / idcg if idcg else 0.0,
            "source_hit": int(all(any(c.get("source") == s for c in ctx) for s in src)),
            "noise_in_context": sum(bool(c.get("noise")) for c in ctx)}


def main(root: Path):
    items = {t["id"]: t for f in (config.TESTSET_FILES, Y2_FILES)
             for t in load_testset([x for x in f if Path(x).exists()]) if t["type"] != "unanswerable"}
    rows = []
    for run, systems in RUNS.items():
        for sys_name, (fname, field) in systems.items():
            ans = load_jsonl(root / run / fname)
            for i, a in ans.items():
                if i in items and a.get(field) is not None:
                    rows.append({"run": run, "system": sys_name, "id": i, "type": items[i]["type"],
                                 **metrics(a[field], items[i])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"Không có câu trả lời nào trong {root}")
    df.to_csv(root / "retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    out = {}
    for (run, sys_name, qt), g in [*df.groupby(["run", "system", "type"]),
                                   *((k + ("all",), g) for k, g in df.groupby(["run", "system"]))]:
        out.setdefault(run, {}).setdefault(sys_name, {})[qt] = {
            "n": len(g), **{m: float(g[m].mean()) for m in ("hit", "recall", "mrr", "ndcg", "source_hit")}}
    (root / "retrieval_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.groupby(["run", "system", "type"])[["hit", "recall", "mrr", "ndcg", "source_hit"]].mean().round(3)
          .to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else config.RESULTS_DIR))
