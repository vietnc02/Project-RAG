"""Kiểm tra judge của ý 2: chấm lại các câu trả lời đã lưu trong results/kg30 với log judge bật.

    python judge_audit.py                  # -> results/kg30_judge_audit/ (judge_audit.md + kết quả chấm lại)
    python judge_audit.py --limit 3        # chạy thử 3 câu (run riêng: results/kg30_judge_audit_thu/)

Điểm gốc của results/kg30 được chấm trước khi có log judge (evaluate.judge nuốt lỗi, trả {}), nên không biết judge
lỗi bao nhiêu lần, có prompt nào bị Ollama cắt không. Script này:
  1. chép answers_*.jsonl của kg30 sang thư mục mới (câu trả lời giữ nguyên, không sinh lại),
  2. chạy evaluate_graphrag.py trên thư mục đó -> chấm lại toàn bộ bằng đúng hàm chấm, có judge_log_<hệ>.jsonl,
  3. tổng hợp: lỗi judge theo loại prompt, độ dài prompt so với num_ctx, và mức tái lập so với điểm gốc
     (cùng judge llama3.1:8b, temperature 0, seed 42; Ollama có thể khác version, xem run_info.json).
"""
import argparse
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import config
from evaluate import judge_log_table, load_jsonl

SRC = config.RESULTS_DIR / "kg30"
SYSTEMS = ("vector_rag", "graphrag")
BINARY = ("hallucination", "answer_correctness", "refusal")
CONTINUOUS = ("context_precision", "faithfulness", "answer_relevancy")


def cohen_kappa(a, b) -> float:
    """Cohen's κ cho hai dãy nhãn (bất kỳ số lớp)."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in labels))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def agreement(out) -> pd.DataFrame:
    rows = []
    for s in SYSTEMS:
        old, new = load_jsonl(SRC / f"scores_{s}.jsonl"), load_jsonl(out / f"scores_{s}.jsonl")
        ids = [i for i in new if i in old]
        for m in (*CONTINUOUS, *BINARY):
            pairs = [(old[i].get(m), new[i].get(m)) for i in ids]
            pairs = [(a, b) for a, b in pairs if not (isnan(a) or isnan(b))]
            if not pairs:
                continue
            a, b = map(np.asarray, zip(*pairs))
            row = {"system": s, "metric": m, "n": len(pairs), "mean_original": round(float(a.mean()), 4),
                   "mean_rejudged": round(float(b.mean()), 4), "exact_agreement": round(float(np.mean(a == b)), 4)}
            if m in BINARY:
                row["cohen_kappa"] = round(cohen_kappa(a, b), 3)
            else:
                row["mean_abs_diff"] = round(float(np.mean(np.abs(a - b))), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def judge_failures(out) -> pd.DataFrame:
    """Judge failures: lần gọi judge không trả đủ verdict hợp lệ (json_error, ollama_error hoặc missing_verdicts).
    Cột step = tên prompt; riêng USEFUL tách thành passages (lần gọi thứ 1 mỗi câu: 4 passage -> retrieval
    precision) và kg_relations (lần thứ 2, GraphRAG: relation của KG -> kg_relation_precision, metric phụ)."""
    rows = []
    for s in SYSTEMS:
        seen = {}
        for line in open(out / f"judge_log_{s}.jsonl", encoding="utf-8"):
            e = json.loads(line)
            step = e["prompt"]
            if step == "useful":
                seen[e["id"]] = seen.get(e["id"], 0) + 1
                step = "passages" if seen[e["id"]] == 1 else "kg_relations"
            if e["status"] != "ok" or (e.get("missing") or 0) > 0:
                rows.append({"system": s, "id": e["id"], "step": step,
                             "failure": e["status"] if e["status"] != "ok" else "missing_verdicts",
                             "missing": e.get("missing"), "expected": e.get("expected"), "error": e.get("error", "")})
    return pd.DataFrame(rows, columns=["system", "id", "step", "failure", "missing", "expected", "error"])


def sensitivity(out, bad_ids: set) -> pd.DataFrame:
    """Sensitivity analysis: paired difference retrieval precision GraphRAG − Vector RAG khi bỏ các câu có judge
    failure ở bước chấm passage."""
    sc = {s: load_jsonl(SRC / f"scores_{s}.jsonl") for s in SYSTEMS}
    types = pd.read_csv(SRC / "per_question.csv").drop_duplicates("id").set_index("id")["type"]
    rows = []
    for name, drop in (("all questions", set()), ("excluding judge failures", bad_ids)):
        for qt in ("all", "single", "multihop"):
            ids = [i for i in sc["graphrag"] if types[i] != "unanswerable" and (qt == "all" or types[i] == qt)
                   and i not in drop]
            d = np.array([sc["graphrag"][i]["context_precision"] - sc["vector_rag"][i]["context_precision"] for i in ids])
            bs = np.random.default_rng(config.SEED).choice(d, size=(5000, len(d))).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"set": name, "type": qt, "n": len(d), "diff": round(float(d.mean()), 4),
                         "lo": round(float(lo), 4), "hi": round(float(hi), 4), "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = config.RESULTS_DIR / ("kg30_judge_audit" + ("_thu" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    for s in SYSTEMS:
        if not (out / f"answers_{s}.jsonl").exists():
            shutil.copy(SRC / f"answers_{s}.jsonl", out / f"answers_{s}.jsonl")

    cmd = [sys.executable, "evaluate_graphrag.py", "--run", out.name, *(["--limit", str(args.limit)] if args.limit else [])]
    if subprocess.run(cmd, cwd=config.PROJECT_DIR).returncode:
        raise SystemExit("evaluate_graphrag.py lỗi")

    logs = [out / f"judge_log_{s}.jsonl" for s in SYSTEMS if (out / f"judge_log_{s}.jsonl").exists()]
    log = judge_log_table(logs)
    agr = agreement(out)
    diff_old = pd.read_csv(SRC / "paired_diff.csv")
    diff_new = pd.read_csv(out / "paired_diff.csv")
    d = diff_old.merge(diff_new, on=["type", "metric"], suffixes=("_original", "_rejudged"))
    d = d[["type", "metric", "diff_original", "ci95_original", "significant_original",
           "diff_rejudged", "ci95_rejudged", "significant_rejudged"]]
    fails = judge_failures(out)
    sens = sensitivity(out, set(fails[fails.step == "passages"].id))
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    info_old = json.loads((SRC / "run_info.json").read_text(encoding="utf-8"))
    tot = log[log.prompt == "(tổng)"]

    lines = ["# Judge audit ý 2: re-judge results/kg30 với judge log\n",
             f"Judge `{config.JUDGE_MODEL}`, num_ctx = {config.NUM_CTX}, num_predict (JSON) = 1024. Ollama gốc "
             f"{info_old['environment']['ollama_server']}, re-judge {info['environment']['ollama_server']} "
             f"(digest model: {info['environment']['models']['judge']['digest']}). Answer giữ nguyên, chỉ chấm lại.\n",
             "## 1. Judge failures và truncate (theo prompt)\n", log.to_markdown(index=False), "",
             f"- Tổng số judge call: {int(tot.calls.sum())}; judge_failures: {int(tot.judge_failures.sum())} "
             f"(json_error: {int(tot.json_error.sum())}, ollama_error: {int(tot.ollama_error.sum())}, missing_verdicts: "
             f"{int(tot.calls_missing_verdicts.sum())} call / {int(tot.verdicts_missing.sum())} verdict); output_cut: "
             f"{int(tot.output_cut.sum())}.",
             f"- Prompt dài nhất: {int(tot.prompt_tokens_max.max())} token (num_ctx = {config.NUM_CTX}); truncated prompt: "
             f"{int(tot.truncated_prompt.sum())}. Ollama truncate prompt dài hơn num_ctx mà không báo lỗi, chỉ giữ đúng "
             f"num_ctx / 2 + 2 = {config.NUM_CTX // 2 + 2} token, nên prompt_eval_count bằng số này (hoặc >= num_ctx) "
             f"được đánh dấu truncated.\n",
             "## 2. Reproducibility (điểm gốc so với re-judge, trên cùng answer)\n", agr.to_markdown(index=False), "",
             "*exact_agreement: tỉ lệ câu có điểm giống hệt; cohen_kappa cho metric nhị phân; mean_abs_diff cho metric "
             "liên tục. faithfulness chỉ tính trên câu không refusal.*\n",
             "## 3. Paired difference GraphRAG − Vector RAG: điểm gốc và re-judge\n", d.to_markdown(index=False), "",
             "## 4. Danh sách judge failures và sensitivity analysis của retrieval precision\n",
             fails.to_markdown(index=False), "",
             "*failure: json_error (JSON hỏng) / ollama_error (Ollama huỷ lần sinh) / missing_verdicts (JSON hợp lệ "
             "nhưng thiếu verdict); verdict thiếu bị tính là \"no\". step passages: chấm 4 passage (retrieval precision); "
             "kg_relations: chấm relation của KG (metric phụ).*\n",
             sens.to_markdown(index=False), "",
             "*Paired difference retrieval precision GraphRAG − Vector RAG, CI 95% bootstrap; \"excluding judge "
             "failures\": bỏ các câu có judge failure ở bước chấm passage của một trong hai hệ.*\n"]
    (out / "judge_audit.md").write_text("\n".join(lines), encoding="utf-8")
    log.to_csv(out / "judge_log_summary.csv", index=False, encoding="utf-8-sig")
    agr.to_csv(out / "judge_agreement.csv", index=False, encoding="utf-8-sig")
    fails.to_csv(out / "judge_failures.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out / "sensitivity.csv", index=False, encoding="utf-8-sig")
    tot_row = log[log.prompt == "(tổng)"]
    (out / "judge_audit.json").write_text(json.dumps({
        "ollama_original": info_old["environment"]["ollama_server"], "ollama_rejudge": info["environment"]["ollama_server"],
        "calls": int(tot_row.calls.sum()), "judge_failures": int(tot_row.judge_failures.sum()),
        "json_error": int(tot_row.json_error.sum()),
        "ollama_error": int(tot_row.ollama_error.sum()), "output_cut": int(tot_row.output_cut.sum()),
        "calls_missing_verdicts": int(tot_row.calls_missing_verdicts.sum()),
        "verdicts_missing": int(tot_row.verdicts_missing.sum()), "truncated": int(tot_row.truncated_prompt.sum()),
        "prompt_tokens_max": int(tot_row.prompt_tokens_max.max()), "num_ctx": config.NUM_CTX,
        "agreement": agr.to_dict(orient="records"), "judge_failure_list": fails.to_dict(orient="records"),
        "sensitivity": sens.to_dict(orient="records")}, ensure_ascii=False, indent=1), encoding="utf-8")
    print((out / "judge_audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
