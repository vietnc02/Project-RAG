"""Tính số liệu + vẽ biểu đồ cho báo cáo nội dung 4 -> report/figures/y4_*.png, report/stats_y4.json

    python report/make_figures_y4.py   (cần results/compression/ từ python evaluate_compression.py)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compress  # noqa: E402
import config  # noqa: E402
from evaluate import judge_log_table, load_jsonl  # noqa: E402
from evaluate_compression import NI_MARGIN, load_items  # noqa: E402

RUN = config.RESULTS_DIR / "compression"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODES = ["base", "kg", "text", "both"]
NAMES = {"base": "Không nén", "kg": "Nén KG", "text": "Nén passage", "both": "Nén KG + passage"}
# base = mốc so sánh -> xám trung tính; ba mức nén = nhóm phân loại -> slot 1, 2, 3 (thứ tự cố định)
C = {"base": "#898781", "kg": "#2a78d6", "text": "#eb6834", "both": "#1baf7a"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})
rng = np.random.default_rng(config.SEED)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def boot(x, n=2000):
    x = np.asarray([v for v in x if not isnan(v)], float)
    if len(x) == 0:
        return [math.nan] * 3
    means = rng.choice(x, size=(n, len(x))).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(x.mean()), float(lo), float(hi)]


def paired(df, metric, mode):
    p = df.pivot(index="id", columns="mode", values=metric)[["base", mode]].dropna()
    d = (p[mode] - p["base"]).to_numpy(float)
    bs = np.random.default_rng(config.SEED).choice(d, size=(5000, len(d))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d),
            "p": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}


# ---------------- số liệu ----------------
items = {t["id"]: t for t in load_items()}
df = pd.read_csv(RUN / "per_question.csv")
ans = {m: load_jsonl(RUN / m / "answers.jsonl") for m in MODES}
df["kg_chars"] = [len(ans[m][i]["kg_context"]) for m, i in zip(df["mode"], df["id"])]
df["passage_chars"] = [sum(len(c["text"]) for c in ans[m][i]["contexts"]) for m, i in zip(df["mode"], df["id"])]
A = df[df.type != "unanswerable"]
U = df[df.type == "unanswerable"]
METRICS = ["faithfulness", "faithfulness_full", "hallucination", "answer_correctness", "context_recall",
           "answer_relevancy", "refusal"]

stats = {"n_questions": int(df.id.nunique()), "n_answerable": int(A.id.nunique()),
         "n_unanswerable": int(U.id.nunique()), "ni_margin": NI_MARGIN,
         "params": {"KG_KEEP": compress.KG_KEEP, "KG_ITEM_CHARS": compress.KG_ITEM_CHARS,
                    "SENT_KEEP": compress.SENT_KEEP},
         "environment": json.loads((RUN / "base" / "run_info.json").read_text(encoding="utf-8"))["environment"],
         "modes": {}, "paired": {}}
for m in MODES:
    g, ga = df[df["mode"] == m], A[A["mode"] == m]
    st = stats["modes"][m] = {
        "context_tokens": float(g.context_tokens.mean()), "prompt_tokens": float(g.prompt_tokens.mean()),
        "kg_chars": float(g.kg_chars.mean()), "passage_chars": float(g.passage_chars.mean()),
        "token_reduction": boot(g.token_reduction.tolist()),
        "token_reduction_min": float(g.token_reduction.min()), "token_reduction_max": float(g.token_reduction.max()),
        "prompt_reduction": float(g.prompt_reduction.mean()),
        "latency_s": float(g.latency_s.mean()),
        "unanswerable_refused": int(U[U["mode"] == m].refusal.sum()),
        "faithfulness_n": int(ga.faithfulness.notna().sum()),
        "refusal_n": int(ga.refusal.sum()),
        "answer_relevancy_non_refused": float(ga[ga.refusal == 0].answer_relevancy.mean())}
    for k in METRICS:
        st[k] = boot(ga[k].tolist())
for m in MODES[1:]:
    stats["paired"][m] = {k: paired(A, k, m) for k in METRICS}
    stats["paired"][m]["token_reduction_target"] = bool(0.19 <= stats["modes"][m]["token_reduction"][0] <= 0.53)
    # năng lực của kiểm định non-inferiority nếu chênh lệch thật = 0: P(ước lượng − 1,96·SE > −margin)
    pf = A.pivot(index="id", columns="mode", values="faithfulness")[["base", m]].dropna()
    se = float((pf[m] - pf["base"]).std(ddof=1) / math.sqrt(len(pf)))
    stats["paired"][m]["ni_power_if_zero"] = 1 - 0.5 * (1 + math.erf(((-NI_MARGIN + 1.96 * se) / se) / math.sqrt(2)))
# faithfulness chỉ trên câu cả 4 chế độ đều trả lời (kiểm tra chọn lọc do refusal)
answered_all = A.pivot(index="id", columns="mode", values="refusal").eq(0).all(axis=1)
A_all = A[A.id.isin(answered_all[answered_all].index)]
stats["faithfulness_answered_all"] = {"n": int(answered_all.sum()),
                                      **{m: paired(A_all, "faithfulness", m) for m in MODES[1:]}}
stats["hallucination_answered"] = {m: float(A[(A["mode"] == m) & (A.refusal == 0)].hallucination.mean())
                                   for m in MODES}
pr_all = [stats["paired"][m][k] for m in MODES[1:] for k in METRICS]
stats["n_comparisons"] = {"total": len(pr_all), "significant": sum(x["lo"] > 0 or x["hi"] < 0 for x in pr_all)}
# câu có đáp án mà bản nén refuse còn bản gốc trả lời
ref = A.pivot(index="id", columns="mode", values="refusal")
cor = A.pivot(index="id", columns="mode", values="answer_correctness")
rec = A.pivot(index="id", columns="mode", values="context_recall")
stats["new_refusals"] = {m: [{"id": i, "type": items[i]["type"], "base_correct": int(cor.loc[i, "base"]),
                              "recall_base": float(rec.loc[i, "base"]), "recall_mode": float(rec.loc[i, m])}
                             for i in ref.index if ref.loc[i, m] == 1 and ref.loc[i, "base"] == 0]
                         for m in MODES[1:]}
stats["correct_flips"] = {m: {"wrong_to_right": int(((cor[m] == 1) & (cor["base"] == 0)).sum()),
                              "right_to_wrong": int(((cor[m] == 0) & (cor["base"] == 1)).sum())} for m in MODES[1:]}
logs = sorted(RUN.glob("*/judge_log.jsonl")) + sorted(RUN.glob("judge_log_gt.jsonl"))
lt = judge_log_table(logs)
tot = lt[lt.prompt == "(tổng)"]
stats["judge_log"] = {"calls": int(tot.calls.sum()), "judge_failures": int(tot.judge_failures.sum()),
                      "json_error": int(tot.json_error.sum()), "ollama_error": int(tot.ollama_error.sum()),
                      "verdicts_missing": int(tot.verdicts_missing.sum()), "truncated": int(tot.truncated_prompt.sum()),
                      "prompt_tokens_max": int(tot.prompt_tokens_max.max())}
# ví dụ minh hoạ: câu có đáp án đầu tiên nhắc tới NIST (context gốc và context nén cả hai phần)
ex_id = next(i for i in ans["base"] if "NIST" in items[i]["question"] and items[i]["type"] != "unanswerable")
b, c = ans["base"][ex_id], ans["both"][ex_id]
kg_lines = lambda t: [x for x in t.splitlines() if x.startswith("- ")]
stats["example"] = {"id": ex_id, "question": items[ex_id]["question"],
                    "kg_base": kg_lines(b["kg_context"]), "kg_both": kg_lines(c["kg_context"]),
                    "passage_base": b["contexts"][0]["text"], "passage_both": c["contexts"][0]["text"],
                    "tokens_base": b["prompt_tokens"] - b["prompt_tokens_no_context"],
                    "tokens_both": c["prompt_tokens"] - c["prompt_tokens_no_context"]}
gt = load_jsonl(RUN / "gt_claims.jsonl")
stats["gt_claims"] = {"n": len(gt), "mean": float(np.mean([len(v["claims"]) for v in gt.values()]))}
(ROOT / "report" / "stats_y4.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ================= Hình 1: token =================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
x = np.arange(len(MODES))
kg = [stats["modes"][m]["kg_chars"] for m in MODES]
ps = [stats["modes"][m]["passage_chars"] for m in MODES]
tok = [stats["modes"][m]["context_tokens"] for m in MODES]
# tách context thành phần KG / passage theo tỉ lệ ký tự, quy về token
kg_tok = [t * k / (k + p) for t, k, p in zip(tok, kg, ps)]
ps_tok = [t - k for t, k in zip(tok, kg_tok)]
ax.bar(x, kg_tok, 0.56, color=[C[m] for m in MODES], edgecolor=SURF, linewidth=2)
ax.bar(x, ps_tok, 0.56, bottom=kg_tok, color=[C[m] for m in MODES], alpha=0.45, edgecolor=SURF, linewidth=2)
for i, m in enumerate(MODES):
    ax.text(i, tok[i] + 25, f"{tok[i]:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(i, kg_tok[i] / 2, "KG", ha="center", va="center", fontsize=8, color="white")
    ax.text(i, kg_tok[i] + ps_tok[i] / 2, "passage", ha="center", va="center", fontsize=8, color=INK)
ax.set_xticks(x, [NAMES[m].replace(" + ", "\n+ ") for m in MODES])
ax.set_ylabel("Token context trung bình")
ax.set_title("(a) Token của context (qwen2.5:7b tokenizer)")
ax.set_ylim(0, max(tok) * 1.15)
grid(ax)

ax = axes[1]
ax.axvspan(19, 53, color=GRID, alpha=0.6, lw=0)
ax.text(36, 2.68, "mục tiêu 19–53%", ha="center", va="bottom", fontsize=8.5, color=INK2)
for j, m in enumerate(MODES[1:]):
    v = df[df["mode"] == m].token_reduction.to_numpy() * 100
    y = len(MODES) - 2 - j
    jitter = np.random.default_rng(j).uniform(-0.18, 0.18, len(v))
    ax.scatter(v, y + jitter, s=12, color=C[m], alpha=0.55, linewidths=0)
    mean, lo, hi = (np.array(stats["modes"][m]["token_reduction"]) * 100)
    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", color=INK, ms=6, capsize=3, zorder=3)
    ax.text(mean, y + 0.32, f"{mean:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_yticks(range(3), [NAMES[m] for m in MODES[1:]][::-1])
ax.set_xlim(0, 70)
ax.set_ylim(-0.6, 3.0)
ax.set_xlabel("Token reduction so với không nén (%), mỗi chấm một câu hỏi")
ax.set_title("(b) Token reduction theo câu hỏi")
grid(ax, "x")
fig.tight_layout()
save(fig, "y4_fig1_tokens.png")

# ================= Hình 2: paired difference so với không nén =================
SHOW = [("faithfulness", "Faithfulness", False), ("context_recall", "Context recall", False),
        ("answer_correctness", "Answer correctness", True), ("refusal", "Refusal\n(câu có đáp án)", True),
        ("answer_relevancy", "Answer relevancy", False)]
fig, axes = plt.subplots(1, len(SHOW), figsize=(13, 3.5), sharey=True)
for ax, (k, title, as_pct) in zip(axes, SHOW):
    s = 100 if as_pct else 1
    ax.axvline(0, color=BASE, lw=1)
    if k == "faithfulness":
        ax.axvline(-NI_MARGIN, color=INK2, lw=1, ls="--")
        ax.text(-NI_MARGIN, 2.6, f" −{NI_MARGIN:g}", ha="left", va="bottom", fontsize=8, color=INK2)
    for j, m in enumerate(MODES[1:]):
        d = stats["paired"][m][k]
        y = 2 - j
        sig = d["lo"] > 0 or d["hi"] < 0
        ax.errorbar(d["delta"] * s, y, xerr=[[(d["delta"] - d["lo"]) * s], [(d["hi"] - d["delta"]) * s]],
                    fmt="o", color=C[m], mfc=C[m] if sig else SURF, mew=1.6, ms=7, capsize=3, lw=1.6)
    ax.set_title(title + ("\n(điểm %)" if as_pct and "\n" not in title else " (điểm %)" if as_pct else ""))
    ax.set_ylim(-0.6, 3.0)
    grid(ax, "x")
axes[0].set_yticks(range(3), [NAMES[m] for m in MODES[1:]][::-1])
fig.text(0.5, -0.05, "Nén − không nén trên cùng câu hỏi (62 câu có đáp án), CI 95% bootstrap; chấm đặc: CI không "
         "chứa 0; đường gạch: ngưỡng non-inferiority của faithfulness", ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout()
save(fig, "y4_fig2_paired.png")
print("saved stats_y4.json + 2 figures")
