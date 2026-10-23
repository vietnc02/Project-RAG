"""Ý 4: context compression cho GraphRAG — token reduction và giữ nguyên faithfulness.

    python evaluate_compression.py                 # 4 chế độ x 85 câu của ý 2 -> results/compression/
    python evaluate_compression.py --limit 3       # chạy thử (results/compression_thu/)
    python evaluate_compression.py --report-only   # chỉ tổng hợp lại từ kết quả đã có

Chế độ (compress.py): base (context GraphRAG gốc), kg (nén mô tả KG), text (nén đoạn văn), both (nén cả hai).
Cả 4 chế độ sinh câu trả lời lại trong cùng một môi trường (cả base), nên chênh lệch chỉ đến từ việc nén. Chạy hết
phần sinh câu trả lời (qwen2.5:7b) trước rồi mới chấm (llama3.1:8b) để Ollama không phải nạp lại model liên tục.

Chỉ số:
  - token reduction  = 1 − token context (nén) / token context (base), theo từng câu; token đếm bằng
                       prompt_eval_count của Ollama (đúng tokenizer của LLM sinh câu trả lời), token context =
                       token prompt − token của cùng prompt với context rỗng
  - faithfulness     = #claim được context ĐƯA VÀO LLM hỗ trợ / #claim (chính); faithfulness_full: chấm trên context
                       gốc chưa nén (phụ: câu trả lời có còn bám vào bằng chứng gốc không); hallucination_rate
  - context_recall   = #ý của đáp án chuẩn suy ra được từ context / #ý ("không mất thông tin quan trọng"); ý của đáp
                       án chuẩn tách 1 lần cho mỗi câu, dùng chung cho mọi chế độ (gt_claims.jsonl)
  - answer_correctness, answer_relevancy, refusal (câu ngoài corpus: đúng = từ chối)
"Giữ nguyên faithfulness" kiểm định non-inferiority, ngưỡng đặt trước NI_MARGIN = 0,05: nén − base có cận dưới
CI 95% (bootstrap ghép cặp) > −0,05 thì kết luận không kém hơn quá 0,05; không thì không đủ bằng chứng.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import compress
import config
from common import env_info, is_refusal
from evaluate import (append_jsonl, bootstrap_ci, flush_judge_log, judge_log_table, load_jsonl, load_testset,
                      ratio, verify)
from evaluate_graphrag import (TESTSET_FILES, answer_correctness, answer_relevancy, extract_claims, gold_sources)
from kg_build import load_papers

MODES = list(compress.MODES)          # base, kg, text, both
ALL_MODES = [*MODES, *compress.BASELINE_MODES]   # + baseline nén ngẫu nhiên / cắt đầu, hai mức nén thêm
NI_MARGIN = 0.05                      # ngưỡng non-inferiority cho faithfulness (đặt trước khi chạy)
METRICS = ["faithfulness", "hallucination", "answer_correctness", "context_recall", "answer_relevancy"]
EXTRA = ["faithfulness_full", "hallucination_full", "refusal", "context_tokens", "prompt_tokens", "latency_s"]


def load_items() -> list[dict]:
    papers = set(load_papers(config.KG_K))
    return [t for t in load_testset([f for f in TESTSET_FILES if f.exists()])
            if t["type"] == "unanswerable" or set(gold_sources(t)) <= papers]


def context_of(a: dict) -> list[dict]:
    """Context đưa vào LLM dưới dạng các khối văn bản (mô tả KG + đoạn văn) để chấm."""
    return [{"text": a["kg_context"]}, *a["contexts"]]


def score(item: dict, a: dict, base: dict, gt_claims: list[str] | None) -> dict:
    q, ans = item["question"], a["answer"]
    refusal = is_refusal(ans)
    s = {"refusal": int(refusal),
         "context_tokens": a["prompt_tokens"] - a["prompt_tokens_no_context"], "prompt_tokens": a["prompt_tokens"]}
    s["context_recall"] = ratio(*verify(gt_claims, context_of(a))) if gt_claims else math.nan
    if refusal:
        s.update(n_claims=0, faithfulness=math.nan, hallucination=0, faithfulness_full=math.nan, hallucination_full=0)
    else:
        claims = extract_claims(q, ans)
        sup, nj = verify(claims, context_of(a))
        sup_f, nj_f = (sup, nj) if a["mode"] == "base" else verify(claims, context_of(base))
        s.update(n_claims=len(claims), n_judged=nj, n_unsupported=nj - sup, faithfulness=ratio(sup, nj),
                 hallucination=int(sup < nj) if nj else math.nan, faithfulness_full=ratio(sup_f, nj_f),
                 hallucination_full=int(sup_f < nj_f) if nj_f else math.nan, claims=claims)
    s["answer_relevancy"] = 0.0 if refusal else answer_relevancy(q, ans)
    s["answer_correctness"] = answer_correctness(item, ans, refusal)
    return s


def paired(df: pd.DataFrame, metric: str, mode: str, n=5000) -> dict | None:
    p = df.pivot(index="id", columns="mode", values=metric)[["base", mode]].dropna()
    if p.empty:
        return None
    d = (p[mode] - p["base"]).to_numpy(float)
    rng = np.random.default_rng(config.SEED)
    bs = rng.choice(d, size=(n, len(d))).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mode": mode, "metric": metric, "n_pairs": len(d), "diff": round(float(d.mean()), 4),
            "ci95": f"[{lo:.3f}, {hi:.3f}]", "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def report(out, items):
    rows = []
    # điểm đã gộp các lượt (judge_passes.py); chưa gộp thì dùng file của lượt hiện tại
    sc_file = lambda m: next((f for f in (out / m / "scores.jsonl", config.pass_path(out / m / "scores.jsonl"))
                              if f.exists()), None)
    modes = [m for m in ALL_MODES if sc_file(m)]
    for m in modes:
        ans, sc = load_jsonl(out / m / "answers.jsonl"), load_jsonl(sc_file(m))
        for t in items:
            if t["id"] in sc:
                rows.append({"mode": m, "id": t["id"], "type": t["type"], "question": t["question"],
                             "answer": ans[t["id"]]["answer"], "latency_s": ans[t["id"]]["latency_s"],
                             "context_chars": ans[t["id"]]["context_chars"],
                             **{k: v for k, v in sc[t["id"]].items() if k not in ("id", "claims")}})
    df = pd.DataFrame(rows)
    base_tok = df[df["mode"] == "base"].set_index("id")["context_tokens"]
    df["token_reduction"] = 1 - df["context_tokens"] / df["id"].map(base_tok)
    df["prompt_reduction"] = 1 - df["prompt_tokens"] / df["id"].map(df[df["mode"] == "base"].set_index("id")["prompt_tokens"])
    df.to_csv(out / "per_question.csv", index=False, encoding="utf-8-sig")

    ans_df = df[df.type != "unanswerable"]
    summ = []
    for m in modes:
        g, ga, gu = df[df["mode"] == m], ans_df[ans_df["mode"] == m], df[(df["mode"] == m) & (df.type == "unanswerable")]
        row = {"mode": m, "n": len(ga)}
        for k in ("token_reduction", "prompt_reduction", *METRICS, "faithfulness_full", "hallucination_full"):
            mean, lo, hi = bootstrap_ci(ga[k].tolist() if k not in ("token_reduction", "prompt_reduction")
                                        else g[k].tolist())
            row[k], row[f"{k}_ci95"] = round(mean, 4), f"[{lo:.3f}, {hi:.3f}]"
        row["context_tokens"] = round(g.context_tokens.mean(), 1)
        row["faithfulness_n"] = int(ga.faithfulness.notna().sum())
        mean, lo, hi = bootstrap_ci(ga.refusal.tolist())   # refuse ở câu CÓ đáp án (hành vi không mong muốn)
        row["refusal_answerable"], row["refusal_answerable_ci95"] = round(mean, 4), f"[{lo:.3f}, {hi:.3f}]"
        row["unanswerable_refused"] = f"{int(gu.refusal.sum())}/{len(gu)}"
        row["latency_s"] = round(g.latency_s.mean(), 2)
        summ.append(row)
    sm = pd.DataFrame(summ)
    pr = pd.DataFrame([x for m in modes[1:] for k in (*METRICS, "faithfulness_full", "refusal")
                       if (x := paired(ans_df, k, m))])
    ni = pr[pr.metric == "faithfulness"].assign(
        margin=-NI_MARGIN, non_inferior=lambda d: d.lo > -NI_MARGIN)[["mode", "n_pairs", "diff", "ci95", "margin",
                                                                       "non_inferior"]]
    sm.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pr.drop(columns=["lo", "hi"]).to_csv(out / "paired_diff.csv", index=False, encoding="utf-8-sig")

    logs = sorted(out.glob("*/judge_log*.jsonl")) + sorted(out.glob("judge_log_gt*.jsonl"))  # + bước tách claim GT
    lines = ["# Kết quả ý 4: context compression cho GraphRAG\n",
             f"{len(items)} câu của ý 2 ({len(ans_df.id.unique())} câu có đáp án). LLM `{config.LLM_MODEL}`, judge "
             f"`{config.JUDGE_MODEL}`. Nén: KG_KEEP = {compress.KG_KEEP}, KG_ITEM_CHARS = {compress.KG_ITEM_CHARS}, "
             f"SENT_KEEP = {compress.SENT_KEEP}. Version: `*/run_info.json`.\n",
             "## 1. Token reduction và chất lượng (câu có đáp án; token reduction tính trên mọi câu)\n",
             sm[["mode", "n", "context_tokens", "token_reduction", "token_reduction_ci95", "prompt_reduction",
                 "faithfulness", "faithfulness_ci95", "faithfulness_n", "hallucination", "answer_correctness",
                 "context_recall", "answer_relevancy", "refusal_answerable", "unanswerable_refused"]]
             .to_markdown(index=False), "",
             "*refusal_answerable: tỉ lệ refuse ở câu CÓ đáp án (câu refusal có answer_relevancy = 0, không có claim "
             "để tính faithfulness).*\n",
             f"## 2. Giữ nguyên faithfulness? (non-inferiority, ngưỡng −{NI_MARGIN})\n", ni.to_markdown(index=False), "",
             "*non_inferior = True: cận dưới CI 95% của (nén − base) > −0,05, tức faithfulness không giảm quá 0,05. "
             "False: không đủ bằng chứng để nói faithfulness được giữ nguyên.*\n",
             "## 3. Chênh lệch ghép cặp so với base (câu có đáp án)\n",
             pr.drop(columns=["lo", "hi"]).to_markdown(index=False), "",
             "## 4. Chỉ số phụ\n", sm[["mode", "faithfulness_full", "faithfulness_full_ci95", "hallucination_full",
                                       "latency_s"]].to_markdown(index=False), "",
             "*faithfulness_full: claim của câu trả lời chấm trên context GỐC chưa nén.*\n"]
    if logs:
        t = pd.concat([judge_log_table([fp]).assign(mode=(fp.parent.name if fp.name.startswith("judge_log.")
                                                          or fp.name == "judge_log.jsonl" else "ground truth")
                                                    + fp.name.removesuffix(".jsonl").removeprefix("judge_log")
                                                    .removeprefix("_gt"))
                       for fp in logs], ignore_index=True)
        lines += ["## 5. Judge failures và truncate\n",t[["mode", *[c for c in t if c not in ("mode", "system")]]].to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out / "summary.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=ALL_MODES)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    out = config.RESULTS_DIR / ("compression" + ("_thu" if args.limit else ""))
    modes = ["base", *[m for m in args.modes if m != "base"]]   # base luôn cần (để so sánh)
    if not args.report_only:
        info = {"started": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
                "compression": {"KG_KEEP": compress.KG_KEEP, "KG_ITEM_CHARS": compress.KG_ITEM_CHARS,
                                "SENT_KEEP": compress.SENT_KEEP},
                "ni_margin": NI_MARGIN, "top_k": config.TOP_K, "num_ctx": config.NUM_CTX,
                "testset": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in TESTSET_FILES if f.exists()},
                "environment": env_info()}
        answers = {}
        for m in modes:   # 1. sinh câu trả lời (qwen2.5:7b) cho mọi chế độ
            (out / m).mkdir(parents=True, exist_ok=True)
            ri = out / m / "run_info.json"
            prev = json.loads(ri.read_text(encoding="utf-8")) if ri.exists() else None
            ri.write_text(json.dumps({**info, "mode": m, "spec": compress.mode_spec(m)} if prev is None else
                                     {**prev, f"judge_pass_{config.JUDGE_PASS}": {"started": info["started"],
                                                                                 "environment": info["environment"]}},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
            answers[m] = load_jsonl(out / m / "answers.jsonl")
            for t in tqdm(items, desc=f"[{m}] trả lời"):
                if t["id"] not in answers[m]:
                    answers[m][t["id"]] = {"id": t["id"], **compress.answer(t["question"], m)}
                    append_jsonl(out / m / "answers.jsonl", answers[m][t["id"]])
        gt_path = out / "gt_claims.jsonl"   # 2. ý của đáp án chuẩn: tách 1 lần, dùng chung
        gts = load_jsonl(gt_path)
        for t in tqdm([t for t in items if t["type"] != "unanswerable"], desc="ý của đáp án chuẩn"):
            if t["id"] not in gts:
                gts[t["id"]] = {"id": t["id"], "claims": extract_claims(t["question"], t["ground_truth"])}
                append_jsonl(gt_path, gts[t["id"]])
                flush_judge_log(out / "judge_log_gt.jsonl", t["id"])
        for m in modes:   # 3. chấm (llama3.1:8b); mỗi lượt chấm một file, judge_passes.py gộp lại
            sp = config.pass_path(out / m / "scores.jsonl")
            sc = load_jsonl(sp)
            for t in tqdm(items, desc=f"[{m}] chấm (lượt {config.JUDGE_PASS})"):
                if t["id"] not in sc:
                    gt = gts[t["id"]]["claims"] if t["id"] in gts else None
                    sc[t["id"]] = {"id": t["id"], **score(t, answers[m][t["id"]], answers["base"][t["id"]], gt)}
                    append_jsonl(sp, sc[t["id"]])
                    flush_judge_log(config.pass_path(out / m / "judge_log.jsonl"), t["id"])
    report(out, items)


if __name__ == "__main__":
    main()
