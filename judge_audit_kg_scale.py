"""Judge audit ý 3: re-judge các câu trả lời GraphRAG với KG10 / KG20 (results/kg_scale/k10, k20) với judge log.

    python judge_audit_kg_scale.py              # -> results/kg_scale_judge_audit/ (judge_audit.md, judge_audit.json)
    python judge_audit_kg_scale.py --limit 3    # chạy thử (results/kg_scale_judge_audit_thu/)

KG30 của ý 3 là kết quả của ý 2 (results/kg30), đã audit bằng judge_audit.py: judge failures của KG30 lấy từ
results/kg30_judge_audit/. Re-judge dùng đúng hàm chấm (evaluate_graphrag.score), câu trả lời giữ nguyên. Tổng hợp:
judge_failures + truncate theo prompt, reproducibility so với điểm gốc, và sensitivity analysis: các paired difference
của báo cáo Y3 (S10, S20, 62 câu) khi bỏ các câu có judge failure ở một trong hai KG được so.
"""
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from evaluate import append_jsonl, flush_judge_log, judge_log_table, load_jsonl
from evaluate_graphrag import score
from evaluate_kg_scale import load_items
from judge_audit import cohen_kappa, isnan

SRC = config.RESULTS_DIR / "kg_scale"
KG30_AUDIT = config.RESULTS_DIR / "kg30_judge_audit"
KS = (10, 20)
METRICS = ("context_precision", "faithfulness", "answer_relevancy", "hallucination", "answer_correctness")
COMPARE = [("S10", 10, 30), ("S10", 20, 30), ("S10", 10, 20), ("S20", 20, 30), ("all", 10, 30), ("all", 20, 30)]


def failures(log_path, k) -> pd.DataFrame:
    """Judge failures (json_error, ollama_error, missing_verdicts) theo câu; USEFUL tách passages / kg_relations."""
    rows, seen = [], {}
    for line in open(log_path, encoding="utf-8"):
        e = json.loads(line)
        step = e["prompt"]
        if step == "useful":
            seen[e["id"]] = seen.get(e["id"], 0) + 1
            step = "passages" if seen[e["id"]] == 1 else "kg_relations"
        if e["status"] != "ok" or (e.get("missing") or 0) > 0:
            rows.append({"k": k, "id": e["id"], "step": step,
                         "failure": e["status"] if e["status"] != "ok" else "missing_verdicts",
                         "missing": e.get("missing"), "expected": e.get("expected"), "error": e.get("error", "")})
    return pd.DataFrame(rows, columns=["k", "id", "step", "failure", "missing", "expected", "error"])


def agreement(k, out) -> list[dict]:
    old, new = load_jsonl(SRC / f"k{k}" / "scores_graphrag.jsonl"), load_jsonl(out / f"k{k}" / "scores_graphrag.jsonl")
    rows = []
    for m in (*METRICS, "refusal"):
        pairs = [(old[i].get(m), new[i].get(m)) for i in new if i in old]
        pairs = [(a, b) for a, b in pairs if not (isnan(a) or isnan(b))]
        if not pairs:
            continue
        a, b = map(np.asarray, zip(*pairs))
        row = {"k": k, "metric": m, "n": len(pairs), "mean_original": round(float(a.mean()), 4),
               "mean_rejudged": round(float(b.mean()), 4), "exact_agreement": round(float(np.mean(a == b)), 4)}
        if m in ("hallucination", "answer_correctness", "refusal"):
            row["cohen_kappa"] = round(cohen_kappa(a, b), 3)
        else:
            row["mean_abs_diff"] = round(float(np.mean(np.abs(a - b))), 4)
        rows.append(row)
    return rows


def determinism(k, out, bad: set) -> dict:
    """Re-judge khác điểm gốc do judge failure hay do judge không tất định: so claim tách ra và số câu khác điểm."""
    old, new = load_jsonl(SRC / f"k{k}" / "scores_graphrag.jsonl"), load_jsonl(out / f"k{k}" / "scores_graphrag.jsonl")
    both = [i for i in new if i in old and old[i].get("claims") and new[i].get("claims")]
    res = {"k": k, "claims_compared": len(both), "claims_identical": sum(old[i]["claims"] == new[i]["claims"] for i in both)}
    for m in ("context_precision", "faithfulness"):
        diff = [i for i in new if i in old and not (isnan(old[i].get(m)) or isnan(new[i].get(m))) and old[i][m] != new[i][m]]
        res[f"{m}_differ"] = len(diff)
        res[f"{m}_differ_with_failure"] = sum(i in bad for i in diff)
    return res


def paired(sc, ids, a, b, m) -> dict | None:
    d = np.array([sc[a][i][m] - sc[b][i][m] for i in ids
                  if not (isnan(sc[a][i].get(m)) or isnan(sc[b][i].get(m)))], float)
    if not len(d):
        return None
    bs = np.random.default_rng(config.SEED).choice(d, size=(5000, len(d))).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(d), "diff": round(float(d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "significant": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = config.RESULTS_DIR / ("kg_scale_judge_audit" + ("_thu" if args.limit else ""))
    items = load_items()
    if args.limit:
        items = items[: args.limit]

    for k in KS:   # 1. re-judge KG10, KG20
        (out / f"k{k}").mkdir(parents=True, exist_ok=True)
        ans = load_jsonl(SRC / f"k{k}" / "answers_graphrag.jsonl")
        sc = load_jsonl(out / f"k{k}" / "scores_graphrag.jsonl")
        for t in tqdm(items, desc=f"[KG{k}] re-judge"):
            if t["id"] not in sc:
                sc[t["id"]] = {"id": t["id"], **score(t, ans[t["id"]])}
                append_jsonl(out / f"k{k}" / "scores_graphrag.jsonl", sc[t["id"]])
                flush_judge_log(out / f"k{k}" / "judge_log_graphrag.jsonl", t["id"])

    # 2. tổng hợp
    logs = [out / f"k{k}" / "judge_log_graphrag.jsonl" for k in KS]
    log = pd.concat([judge_log_table([fp]).assign(k=f"k{k}") for fp, k in zip(logs, KS)], ignore_index=True)
    fails = pd.concat([failures(fp, k) for fp, k in zip(logs, KS)]
                      + [failures(KG30_AUDIT / "judge_log_graphrag.jsonl", 30)], ignore_index=True)
    agr = pd.DataFrame([r for k in KS for r in agreement(k, out)])
    det = pd.DataFrame([determinism(k, out, set(fails[fails.k == k].id)) for k in KS])

    # 3. sensitivity analysis trên các paired difference của báo cáo Y3 (điểm gốc)
    sc = {k: load_jsonl(SRC / f"k{k}" / "scores_graphrag.jsonl") for k in (10, 20, 30)}
    scope = {t["id"]: t["scope"] for t in items}
    sets = {"S10": [i for i, s in scope.items() if s == 10], "S20": [i for i, s in scope.items() if 0 < s <= 20],
            "all": [i for i, s in scope.items() if s > 0]}
    bad = {k: set(fails[fails.k == k].id) for k in (10, 20, 30)}
    sens = []
    for name, a, b in COMPARE:
        for m in METRICS:
            full = paired(sc, sets[name], a, b, m)
            excl = paired(sc, [i for i in sets[name] if i not in bad[a] | bad[b]], a, b, m)
            if full and excl:
                sens.append({"set": name, "compare": f"KG{a} − KG{b}", "metric": m,
                             **{f"{x}_all": full[x] for x in ("n", "diff", "lo", "hi", "significant")},
                             **{f"{x}_excl": excl[x] for x in ("n", "diff", "lo", "hi", "significant")}})
    # điểm re-judge (KG10, KG20 chấm lại; KG30 = re-judge của ý 2) -> paired difference có giữ kết luận không
    sc_re = {10: load_jsonl(out / "k10" / "scores_graphrag.jsonl"), 20: load_jsonl(out / "k20" / "scores_graphrag.jsonl"),
             30: load_jsonl(KG30_AUDIT / "scores_graphrag.jsonl")}
    for row in sens:
        name, a, b = row["set"], int(row["compare"].split()[0][2:]), int(row["compare"].split()[-1][2:])
        re = paired(sc_re, sets[name], a, b, row["metric"])
        row.update({f"{x}_rejudged": re[x] for x in ("diff", "lo", "hi", "significant")})
    sens = pd.DataFrame(sens)
    flips = sens[(sens.significant_all != sens.significant_excl) | (sens.significant_all != sens.significant_rejudged)]

    tot = log[log.prompt == "(tổng)"]
    lines = ["# Judge audit ý 3: re-judge GraphRAG với KG10 / KG20 (KG30: results/kg30_judge_audit)\n",
             f"Judge `{config.JUDGE_MODEL}`, num_ctx = {config.NUM_CTX}. Answer giữ nguyên, chỉ chấm lại.\n",
             "## 1. Judge failures và truncate (theo prompt)\n", log.to_markdown(index=False), "",
             f"- Tổng (KG10 + KG20): {int(tot.calls.sum())} judge call, judge_failures {int(tot.judge_failures.sum())}, "
             f"truncated prompt {int(tot.truncated_prompt.sum())}, prompt dài nhất {int(tot.prompt_tokens_max.max())} token.\n",
             "## 2. Danh sách judge failures (KG10, KG20 và KG30)\n", fails.to_markdown(index=False), "",
             "## 3. Reproducibility (điểm gốc so với re-judge)\n", agr.to_markdown(index=False), "",
             det.to_markdown(index=False), "",
             "*claims_identical: số câu mà judge tách ra đúng cùng danh sách claim ở hai lần chấm; *_differ: số câu có "
             "điểm khác nhau, *_differ_with_failure: trong đó có judge failure khi re-judge. Cùng Ollama 0.35.1 / digest "
             "model ở hai lần chấm gốc của ý 2 và ý 3, nhưng chỉ ý 2 tái lập y hệt.*\n",
             "## 4. Sensitivity analysis: paired difference của Y3 khi bỏ các câu có judge failure\n",
             sens.to_markdown(index=False), "",
             f"*_all: điểm gốc, đủ câu; _excl: điểm gốc, bỏ các câu có judge failure ở một trong hai KG được so; "
             f"_rejudged: điểm re-judge, đủ câu. Số so sánh đổi kết luận (significant) ở ít nhất một cách: {len(flips)}.*\n",
             "### Các so sánh đổi kết luận\n", flips.to_markdown(index=False), ""]
    (out / "judge_audit.md").write_text("\n".join(lines), encoding="utf-8")
    fails.to_csv(out / "judge_failures.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out / "sensitivity.csv", index=False, encoding="utf-8-sig")
    (out / "judge_audit.json").write_text(json.dumps({
        "calls": int(tot.calls.sum()), "judge_failures": int(tot.judge_failures.sum()),
        "json_error": int(tot.json_error.sum()), "ollama_error": int(tot.ollama_error.sum()),
        "verdicts_missing": int(tot.verdicts_missing.sum()), "truncated": int(tot.truncated_prompt.sum()),
        "prompt_tokens_max": int(tot.prompt_tokens_max.max()), "num_ctx": config.NUM_CTX,
        "judge_failure_list": fails.to_dict(orient="records"), "agreement": agr.to_dict(orient="records"),
        "determinism": det.to_dict(orient="records"),
        "sensitivity": sens.to_dict(orient="records"), "flips": flips.to_dict(orient="records")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print((out / "judge_audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
