"""Tính số liệu + vẽ biểu đồ cho báo cáo nội dung 1 -> report/figures/*.png, report/stats.json

Đọc <RESULTS_DIR>/k4 (top-k = 4, cả 3 loại câu hỏi), k8 (top-k = 8) và noise4 (top-4 + 4 đoạn ngẫu nhiên của bài
khác), closed_book (baseline không truy xuất), retrieval_metrics.json, threshold_refusal*.
    python report/make_figures.py
"""
import json
import math
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import pandas as pd  # noqa: E402
from evaluate import judge_log_table, load_jsonl  # noqa: E402
from stat_utils import boot, holm, isnan, outcomes, paired_boot  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette: reference categorical slots 1-2 (light mode), chrome & ink ----
C = {"lc": "#2a78d6", "sr": "#eb6834"}
C_LIGHT = {"lc": "#a9c8ef", "sr": "#f6c0a9"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
NAMES = {"lc": "LangChain RAG", "sr": "LangGraph Self-RAG"}
SYS = {"lc": "langchain_rag", "sr": "langgraph_selfrag"}
TYPES = ["single", "multihop", "unanswerable"]
TYPE_VI = {"single": "Single-hop", "multihop": "Multi-hop", "unanswerable": "Out-of-corpus"}
METRICS = [("faithfulness", "Faithfulness", "↑", False),
           ("answer_relevancy", "Answer relevancy", "↑", False),
           ("hallucination", "Hallucination rate", "↓", True)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)




# ---------------- load ----------------
TS = {}
for f in config.TESTSET_FILES:
    for t in json.loads(Path(f).read_text(encoding="utf-8")):
        if t.get("keep", True) and t.get("reviewed"):
            TS[t["id"]] = {"type": "single", **t}

RUNS = {}
for run in ("k4", "k8", "noise4"):
    d = config.RESULTS_DIR / run
    if not d.exists():
        continue
    A = {s: load_jsonl(d / f"answers_{SYS[s]}.jsonl") for s in SYS}
    S = {s: load_jsonl(d / f"scores_{SYS[s]}.jsonl") for s in SYS}
    ids = sorted(i for i in S["lc"] if i in S["sr"] and i in TS)
    if not ids:  # run chưa có kết quả của cả 2 hệ
        continue
    RUNS[run] = {"A": A, "S": S, "ids": ids,
                 "info": json.loads((d / "run_info.json").read_text(encoding="utf-8"))}


def ids_of(run, qtype):
    return [i for i in RUNS[run]["ids"] if TS[i]["type"] == qtype]


# ---------------- stats ----------------
stats = {"runs": {}, "testset": {}, "corpus": {}}
for run, R in RUNS.items():
    A, S = R["A"], R["S"]
    rs = stats["runs"][run] = {"top_k": R["info"]["top_k"], "environment": R["info"]["environment"], "types": {}}
    for qt in TYPES:
        ids = ids_of(run, qt)
        if not ids:
            continue
        ts = rs["types"][qt] = {"n": len(ids), "main": {}, "n_metric": {}, "extra": {}, "paired": {}}
        for s in SYS:
            ts["main"][s] = {m: boot([S[s][i][m] for i in ids]) for m, *_ in METRICS}
            # số câu thực sự được tính (faithfulness bỏ câu refusal, hallucination thì không)
            ts["n_metric"][s] = {m: sum(not isnan(S[s][i].get(m)) for i in ids) for m, *_ in METRICS}
            answered = [i for i in ids if not S[s][i]["refusal"]]
            nc = sum(S[s][i].get("n_judged", S[s][i]["n_claims"]) for i in ids)
            ts["extra"][s] = {
                "faithfulness_used": boot([S[s][i]["faithfulness_used"] for i in ids]),
                "hallucination_used": float(np.mean([S[s][i]["hallucination_used"] for i in ids])),
                "claim_halluc_rate": sum(S[s][i]["n_unsupported"] for i in ids) / nc if nc else math.nan,
                "halluc_among_answered": (float(np.mean([S[s][i]["hallucination"] for i in answered]))
                                          if answered else math.nan),
                "ar_among_answered": (float(np.mean([S[s][i]["answer_relevancy"] for i in answered]))
                                      if answered else math.nan),
                "n_answered": len(answered),
                "claims_per_answer": (float(np.mean([S[s][i]["n_claims"] for i in answered])) if answered else math.nan),
                "refusal_rate": float(np.mean([S[s][i]["refusal"] for i in ids])),
                "answer_correctness": float(np.mean([S[s][i]["answer_correctness"] for i in ids])),
                "retrieval_hit": (float(np.nanmean([S[s][i]["retrieval_hit"] for i in ids]))
                                  if qt != "unanswerable" else math.nan),
                "latency_s": float(np.mean([A[s][i]["latency_s"] for i in ids])),
                "llm_calls": float(np.mean([A[s][i]["llm_calls"] for i in ids])),
            }
        # paired difference (Self-RAG − LangChain RAG) trên cùng câu hỏi
        for m, *_ in METRICS:
            d = np.array([S["sr"][i][m] - S["lc"][i][m] for i in ids
                          if not (isnan(S["sr"][i][m]) or isnan(S["lc"][i][m]))], float)
            ts["paired"][m] = paired_boot(d)
        # nhóm kết cục: tách hallucination khỏi refusal (hallucination / correctness trên câu có trả lời)
        ts["outcomes"] = {s: outcomes([(qt != "unanswerable", S[s][i]["refusal"], S[s][i]["answer_correctness"],
                                        S[s][i]["hallucination"]) for i in ids]) for s in SYS}
        h = Counter((S["lc"][i]["hallucination"], S["sr"][i]["hallucination"]) for i in ids)
        b_, c_ = h[(1, 0)], h[(0, 1)]
        ts["mcnemar"] = {"both": h[(1, 1)], "only_lc": b_, "only_sr": c_, "neither": h[(0, 0)],
                         "p": (min(1.0, 2 * sum(comb(b_ + c_, k) for k in range(min(b_, c_) + 1)) / 2 ** (b_ + c_))
                               if b_ + c_ else 1.0)}
        ends = Counter()
        for i in ids:
            t = A["sr"][i]["trace"][-1]
            ends["abstain" if t == "abstain" else t.split("(")[-1].rstrip(")")] += 1
        ts["sr_behaviour"] = {"ends": dict(ends),
                              "rewrites": sum(A["sr"][i].get("rewrites", 0) > 0 for i in ids),
                              "regenerations": sum(A["sr"][i].get("regenerations", 0) > 0 for i in ids),
                              "docs_kept": float(np.mean([len(A["sr"][i]["contexts"]) for i in ids]))}

    # hiệu chỉnh Holm: họ so sánh = mọi paired difference của lần chạy này (một bảng trong báo cáo)
    rs["holm_family"] = holm([ts["paired"][m] for ts in rs["types"].values() for m, *_ in METRICS])

    # abstain đúng/sai: câu out-of-corpus phải refuse; câu có đáp án (single + multi-hop) không được refuse
    ab = rs["abstention"] = {}
    for s in SYS:
        unans = [i for i in R["ids"] if TS[i]["type"] == "unanswerable"]
        answ = [i for i in R["ids"] if TS[i]["type"] != "unanswerable"]
        ab[s] = {"unans_abstain": sum(S[s][i]["refusal"] for i in unans), "unans_n": len(unans),
                 "answ_abstain": sum(S[s][i]["refusal"] for i in answ), "answ_n": len(answ)}

# ---- Self-RAG có chọn câu dễ không (Y1.4): trên đúng các câu multi-hop mà cả hai hệ cùng trả lời ----
stats["selection"] = {}
for run, R in RUNS.items():
    S = R["S"]
    for qt in ("single", "multihop"):
        ids = ids_of(run, qt)
        both = [i for i in ids if not S["sr"][i]["refusal"] and not S["lc"][i]["refusal"]]
        sr_ref = [i for i in ids if S["sr"][i]["refusal"] and not S["lc"][i]["refusal"]]
        d = [S["sr"][i]["hallucination"] - S["lc"][i]["hallucination"] for i in both]
        stats["selection"].setdefault(run, {})[qt] = {
            "n_both": len(both), "sr_halluc": float(np.mean([S["sr"][i]["hallucination"] for i in both])) if both else math.nan,
            "lc_halluc": float(np.mean([S["lc"][i]["hallucination"] for i in both])) if both else math.nan,
            "paired": paired_boot(d) if both else None, "n_sr_refused_lc_answered": len(sr_ref),
            "lc_halluc_on_sr_refused": (float(np.mean([S["lc"][i]["hallucination"] for i in sr_ref]))
                                        if sr_ref else math.nan)}

# ---- baseline không truy xuất (closed-book) ----
cb = load_jsonl(config.RESULTS_DIR / "closed_book" / "scores.jsonl")
if cb and "k4" in RUNS:
    S4 = RUNS["k4"]["S"]
    stats["closed_book"] = {}
    for qt in TYPES:
        ids = [i for i in ids_of("k4", qt) if i in cb]
        cu = [i for i in ids if not S4["lc"][i]["refusal"] and S4["lc"][i]["answer_correctness"]
              and S4["lc"][i]["hallucination"]]
        stats["closed_book"][qt] = {
            "n": len(ids), "correct": float(np.mean([cb[i]["answer_correctness"] for i in ids])),
            "refusal": float(np.mean([cb[i]["refusal"] for i in ids])),
            "n_lc_correct_unsupported": len(cu),
            "cb_correct_among_lc_correct_unsupported": (float(np.mean([cb[i]["answer_correctness"] for i in cu]))
                                                        if cu else math.nan)}

# ---- chỉ số truy xuất không qua judge (gold chunk) và baseline refuse theo ngưỡng cosine ----
for key, fn in (("retrieval", "retrieval_metrics.json"), ("threshold", "threshold_refusal.json")):
    fp = config.RESULTS_DIR / fn
    if fp.exists():
        stats[key] = json.loads(fp.read_text(encoding="utf-8"))
THR_CURVE = config.RESULTS_DIR / "threshold_refusal_curve.csv"

# ---- độ tin cậy của judge: 3 lần chấm (bảng chính top-k = 4) và thử lại khi judge lỗi ----
jp = config.RESULTS_DIR / "k4" / "judge_passes.json"
if jp.exists():
    stats["judge_passes"] = {s: v for s, v in json.loads(jp.read_text(encoding="utf-8")).items()}
logs = sorted(p for run in RUNS for p in (config.RESULTS_DIR / run).glob("judge_log_*.jsonl"))
if logs:
    lt = judge_log_table(logs)
    tot = lt[lt.prompt == "(tổng)"]
    stats["judge_log"] = {k: int(tot[k].sum()) for k in ("calls", "judge_failures", "retries", "unresolved",
                                                         "truncated_prompt")}

raw = {"single": "testset_raw.json", "multihop": "multihop_raw.json"}
for qt in TYPES:
    kept = sum(1 for t in TS.values() if t["type"] == qt)
    gen = (len(json.loads((ROOT / "testset" / raw[qt]).read_text(encoding="utf-8"))) if qt in raw else kept)
    stats["testset"][qt] = {"generated": gen, "kept": kept}
papers, chunks = defaultdict(set), Counter()
with open(config.CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        papers[c["topic"]].add(c["source"])
        chunks[c["topic"]] += 1
stats["corpus"] = {tp: {"papers": len(p), "chunks": chunks[tp]} for tp, p in papers.items()}
# Chấm tay giai đoạn 2 (human_audit.py --score): agreement người chấm – judge trên nội dung 1–4
HA = config.RESULTS_DIR / "human_audit" / "stats.json"
if HA.exists():
    stats["human_audit"] = json.loads(HA.read_text(encoding="utf-8"))
(ROOT / "report" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


# ================= FIGURES =================
def bar_ci(ax, x, v, color, pct, w=0.36, hatch=None, label=None):
    mean, lo, hi = v
    if isnan(mean):
        return
    f = 100 if pct else 1
    ax.bar(x, mean * f, w, color=color, edgecolor=SURF, lw=2, hatch=hatch, label=label)
    ax.errorbar(x, mean * f, yerr=[[(mean - lo) * f], [(hi - mean) * f]], color=INK, capsize=3, lw=1)
    ax.text(x, hi * f + (1.5 if pct else 0.015), f"{mean * f:.1f}%" if pct else f"{mean:.2f}",
            ha="center", va="bottom", fontsize=7.5, color=INK2)


# Fig 1 - architecture
def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=INK)


def arrow(ax, x1, y1, x2, y2, label=None, rad=0.0, lx=0, ly=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.5, color=INK2, ha="center",
                va="center", bbox=dict(fc=SURF, ec="none", pad=0.5))


calls = (stats["runs"].get("k4", {}).get("types", {}).get("single", {}).get("extra", {})
         .get("sr", {}).get("llm_calls"))
fig, axes = plt.subplots(2, 1, figsize=(9, 7.4), gridspec_kw={"height_ratios": [1, 2.4]})
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
ax.set_title("(a) LangChain RAG thuần: 1 lần gọi LLM", loc="left")
xs = [1.0, 3.4, 5.8, 8.4]
for x, t in zip(xs, ["Câu hỏi", "Retriever\n(Chroma, top-k)", "Generate\n(qwen2.5:7b)", "Câu trả lời"]):
    box(ax, x, 1.0, 1.8, 0.9, t, fc="#eaf2fc" if ("Retr" in t or "Gen" in t) else "#ffffff", ec=C["lc"])
for a, b in zip(xs, xs[1:]):
    arrow(ax, a + 0.9, 1.0, b - 0.9, 1.0)
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
ax.set_title("(b) LangGraph Self-RAG"
             + (f" (trung bình {calls:.1f} lần gọi LLM)" if calls else ""), loc="left")
P = {"q": (0.8, 4.6), "ret": (2.8, 4.6), "grade": (5.2, 4.6), "gen": (7.6, 4.6), "gg": (7.6, 2.5),
     "end": (8.2, 0.6), "tq": (2.8, 2.5), "abs": (5.2, 0.6), "regen": (9.3, 3.55)}
fcs = "#fdeee7"
box(ax, *P["q"], 1.3, 0.8, "Câu hỏi")
box(ax, *P["ret"], 1.7, 0.8, "retrieve\n(top-k)", fc=fcs, ec=C["sr"])
box(ax, *P["grade"], 2.1, 0.8, "grade_documents\n(LLM chấm liên quan)", fc=fcs, ec=C["sr"])
box(ax, *P["gen"], 1.5, 0.8, "generate", fc=fcs, ec=C["sr"])
box(ax, *P["gg"], 2.4, 0.9, "grade_generation (LLM chấm)\n1) grounded? 2) useful?", fc=fcs, ec=C["sr"])
box(ax, *P["tq"], 1.9, 0.8, "transform_query\n(viết lại truy vấn)", fc=fcs, ec=C["sr"])
box(ax, *P["regen"], 1.3, 0.8, "regenerate\n(chặt hơn)", fc=fcs, ec=C["sr"])
box(ax, *P["abs"], 1.9, 0.8, "abstain\n(refuse trả lời)")
box(ax, *P["end"], 1.6, 0.8, "Câu trả lời")
arrow(ax, 1.45, 4.6, 1.95, 4.6)
arrow(ax, 3.65, 4.6, 4.15, 4.6)
arrow(ax, 6.25, 4.6, 6.85, 4.6, "có tài liệu\nliên quan", ly=0.68)
arrow(ax, 7.6, 4.2, 7.6, 2.95)
arrow(ax, 8.2, 2.05, 8.2, 1.0, "grounded\n& useful", lx=0.55)
arrow(ax, 8.8, 2.75, 9.1, 3.15, "không grounded\n(còn lượt)", lx=0.55, ly=-0.55)
arrow(ax, 9.1, 3.95, 8.35, 4.45, rad=0.25)
arrow(ax, 6.4, 2.5, 3.75, 2.5, "không useful (còn lượt)", lx=-0.55, ly=0.22)
arrow(ax, 4.6, 4.2, 3.4, 2.9, "0 tài liệu\n(còn lượt)", lx=-0.45, ly=0.1)
arrow(ax, 2.8, 2.9, 2.8, 4.2)
arrow(ax, 5.35, 4.2, 5.35, 1.0, "0 tài liệu\n(hết lượt)", ly=0.55)
arrow(ax, 7.0, 2.05, 6.15, 0.8, "không grounded\n(hết lượt)", lx=-0.15, ly=-0.05)
fig.tight_layout()
save(fig, "fig1_architecture.png")

# Fig 2 - testset composition
fig, ax = plt.subplots(figsize=(8, 2.4))
for y, qt in enumerate(reversed(TYPES)):
    g, k = stats["testset"][qt]["generated"], stats["testset"][qt]["kept"]
    ax.barh(y, g, color="#e8e7e1", height=0.55, edgecolor=SURF, lw=2)
    ax.barh(y, k, color=C["lc"], height=0.55, edgecolor=SURF, lw=2)
    note = "viết tay" if qt == "unanswerable" else f"sinh tự động {g} → giữ {k} sau duyệt tay"
    ax.text(g + 1.5, y, f"{k} câu ({note})", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(3)); ax.set_yticklabels([TYPE_VI[t] for t in reversed(TYPES)])
ax.set_xlim(0, 150); ax.set_xlabel("Số câu hỏi"); grid(ax, "x")
ax.set_title(f"Bộ câu hỏi đánh giá: {len(TS)} câu", loc="left")
save(fig, "fig2_testset.png")


# Fig 3 - three metrics by question type (run k4)
def fig_main(run, name):
    rs = stats["runs"][run]["types"]
    types = [t for t in ("single", "multihop") if t in rs]  # câu out-of-corpus: không có claim -> trình bày bằng bảng
    fig, axes = plt.subplots(len(types), 3, figsize=(10, 2.6 * len(types)), squeeze=False)
    for r, qt in enumerate(types):
        for c_, (m, title, arrow_, pct) in enumerate(METRICS):
            ax = axes[r][c_]
            for k, s in enumerate(SYS):
                bar_ci(ax, k, rs[qt]["main"][s][m], C[s], pct, w=0.6)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["LangChain\nRAG", "Self-RAG"], fontsize=8.5)
            ax.set_ylim(0, 105 if pct else 1.12)
            if pct:
                ax.set_ylabel("% câu hỏi", fontsize=8.5)
            ax.set_title(f"{title} {arrow_}" if r == 0 else "", fontsize=10)
            if c_ == 0:
                ax.text(-0.42, 0.5, f"{TYPE_VI[qt]}\n(n = {rs[qt]['n']})", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
            grid(ax)
    fig.suptitle(f"Ba chỉ số theo loại câu hỏi, top-k = {stats['runs'][run]['top_k']} "
                 "(thanh lỗi = CI 95% bootstrap)", x=0.01, ha="left", fontsize=11, fontweight="bold",
                 color=INK)
    fig.tight_layout()
    save(fig, name)


for run in RUNS:
    fig_main(run, f"fig3_main_{run}.png")

# Fig 4 - paired differences (forest), all conditions
rows = [(run, qt) for run in RUNS for qt in ("single", "multihop") if qt in stats["runs"][run]["types"]]
fig, axes = plt.subplots(1, 3, figsize=(11, 0.55 * len(rows) + 1.4), sharey=True)
for ax, (m, title, arrow_, pct) in zip(axes, METRICS):
    for y, (run, qt) in enumerate(reversed(rows)):
        p = stats["runs"][run]["types"][qt]["paired"][m]
        if p is None:
            continue
        sig = not (p["lo"] <= 0 <= p["hi"])
        col = C["sr"] if sig else MUTED
        f = 100 if pct else 1
        ax.plot([p["lo"] * f, p["hi"] * f], [y, y], color=col, lw=2)
        ax.plot(p["delta"] * f, y, "o", color=col, ms=8, mec=SURF, mew=1.5)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{TYPE_VI[qt]}, top-k={stats['runs'][run]['top_k']}" for run, qt in reversed(rows)],
                       fontsize=8.5)
    ax.set_title(f"{title} {arrow_}", fontsize=10)
    ax.set_xlabel("Δ điểm %" if pct else "Δ điểm", fontsize=8.5)
    grid(ax, "x")
fig.suptitle("Paired difference Self-RAG − LangChain RAG trên cùng câu hỏi "
             "(cam = CI 95% không chứa 0)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "fig4_paired.png")

# Fig 5 - ba điều kiện truy xuất: top-4, top-8, top-4 + 4 đoạn nhiễu ngẫu nhiên
COND = [r for r in ("k4", "k8", "noise4") if r in RUNS]
COND_LABEL = {"k4": "top-4", "k8": "top-8", "noise4": "+nhiễu"}
HATCH = {"k4": None, "k8": "//", "noise4": ".."}
if len(COND) > 1:
    types = [t for t in ("single", "multihop") if all(t in stats["runs"][r]["types"] for r in COND)]
    fig, axes = plt.subplots(len(types), 3, figsize=(11, 2.8 * len(types)), squeeze=False)
    nc_ = len(COND)
    for r, qt in enumerate(types):
        for c_, (m, title, arrow_, pct) in enumerate(METRICS):
            ax = axes[r][c_]
            xs = []
            for si, s in enumerate(("lc", "sr")):
                for k, run in enumerate(COND):
                    x = si * (nc_ + 0.5) + k
                    xs.append(x)
                    bar_ci(ax, x, stats["runs"][run]["types"][qt]["main"][s][m],
                           C[s] if run == "k4" else C_LIGHT[s], pct, w=0.78, hatch=HATCH[run])
            ax.set_xticks(xs); ax.set_xticklabels([COND_LABEL[r] for r in COND] * 2, fontsize=7.5)
            ax.text((nc_ - 1) / 2, -0.27, "LangChain RAG", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=8, color=C["lc"], fontweight="bold")
            ax.text(nc_ + 0.5 + (nc_ - 1) / 2, -0.27, "Self-RAG", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=8, color=C["sr"], fontweight="bold")
            ax.set_ylim(0, 105 if pct else 1.12)
            ax.set_title(f"{title} {arrow_}" if r == 0 else "", fontsize=10)
            if c_ == 0:
                ax.text(-0.4, 0.5, TYPE_VI[qt], transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=INK)
            grid(ax)
    fig.suptitle("Ba điều kiện truy xuất: top-4, top-8 (gạch chéo) và top-4 + 4 đoạn nhiễu ngẫu nhiên (chấm)",
                 x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(h_pad=2.2)
    save(fig, "fig5_noisy.png")

# Fig 6 - scoring context: full top-k vs filtered docs (Self-RAG, run k4)
if "k4" in RUNS:
    rs = stats["runs"]["k4"]["types"]
    types = [t for t in ("single", "multihop") if t in rs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    x = np.arange(len(types)); w = 0.36
    for k, (key, lab, col) in enumerate([("main", "Full top-k ban đầu (chung cho 2 hệ)", C["sr"]),
                                         ("used", "Chỉ tài liệu đã lọc", C_LIGHT["sr"])]):
        fv = [rs[t]["main"]["sr"]["faithfulness"][0] if key == "main" else rs[t]["extra"]["sr"]["faithfulness_used"][0]
              for t in types]
        hv = [rs[t]["main"]["sr"]["hallucination"][0] if key == "main" else rs[t]["extra"]["sr"]["hallucination_used"]
              for t in types]
        for ax, vals, pct in ((axes[0], fv, False), (axes[1], hv, True)):
            v = [vv * (100 if pct else 1) for vv in vals]
            ax.bar(x + (k - 0.5) * w, v, w, color=col, edgecolor=SURF, lw=2, label=lab)
            for xi, vi in zip(x + (k - 0.5) * w, v):
                ax.text(xi, vi + (1 if pct else 0.01), f"{vi:.1f}%" if pct else f"{vi:.2f}", ha="center",
                        fontsize=7.5, color=INK2)
    for ax, t_, pct in ((axes[0], "Faithfulness của Self-RAG ↑", False), (axes[1], "Hallucination rate của Self-RAG ↓", True)):
        ax.set_xticks(x); ax.set_xticklabels([TYPE_VI[t] for t in types]); ax.set_title(t_, loc="left")
        ax.set_ylim(0, 100 if pct else 1.1); grid(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig6_scoring_context.png")

# Fig 7 - secondary: refusal / correctness / cost (run k4)
if "k4" in RUNS:
    rs = stats["runs"]["k4"]["types"]
    types = [t for t in TYPES if t in rs]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    x = np.arange(len(types)); w = 0.36
    for ax, (key, title, pct) in zip(axes, [("refusal_rate", "Refusal rate trả lời", True),
                                            ("answer_correctness", "Đúng so với ground truth*", True),
                                            ("latency_s", "Thời gian (giây / câu)", False)]):
        for k, s in enumerate(SYS):
            v = [rs[t]["extra"][s][key] * (100 if pct else 1) for t in types]
            ax.bar(x + (k - 0.5) * w, v, w, color=C[s], edgecolor=SURF, lw=2, label=NAMES[s])
            for xi, vi in zip(x + (k - 0.5) * w, v):
                ax.text(xi, vi + (1.5 if pct else 0.5), f"{vi:.0f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([TYPE_VI[t] for t in types], fontsize=8.5)
        ax.set_title(title, loc="left"); grid(ax)
        ax.set_ylim(0, 110 if pct else None)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, -0.03, "* Câu out-of-corpus: đúng = refusal trả lời.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig7_secondary.png")

    # Fig 8 - Self-RAG behaviour (run k4)
    lab = {"useful": "Qua kiểm tra (grounded & useful)", "abstain": "Abstain (refuse)",
           "not_useful": "Hết lượt, chưa useful", "refusal": "Model tự trả lời 'không tìm thấy'"}
    fig, axes = plt.subplots(1, len(types), figsize=(11, 2.6), sharex=True)
    for ax, qt in zip(np.atleast_1d(axes), types):
        e = rs[qt]["sr_behaviour"]["ends"]
        items = [(k, e.get(k, 0)) for k in ("useful", "not_useful", "refusal", "abstain")]
        for y, (k, v) in enumerate(reversed(items)):
            ax.barh(y, v, color=C["sr"] if k == "useful" else C_LIGHT["sr"], height=0.6, edgecolor=SURF, lw=2)
            ax.text(v + 0.8, y, str(v), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(4)); ax.set_yticklabels([lab[k] for k, _ in reversed(items)], fontsize=8)
        ax.set_title(f"{TYPE_VI[qt]} (n = {rs[qt]['n']})", loc="left", fontsize=9.5); grid(ax, "x")
        if ax is not np.atleast_1d(axes)[0]:
            ax.set_yticklabels([])
    fig.suptitle("Trạng thái kết thúc của Self-RAG, top-k = 4", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig8_selfrag_behaviour.png")

# Fig 9 - corpus
cp = sorted(stats["corpus"].items(), key=lambda kv: kv[1]["papers"])
fig, ax = plt.subplots(figsize=(8, 3.2))
for yy, (tp, d) in enumerate(cp):
    ax.barh(yy, d["papers"], color=C["lc"], height=0.6, edgecolor=SURF, lw=2)
    ax.text(d["papers"] + 0.4, yy, f"{d['papers']} bài · {d['chunks']} chunk", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(cp))); ax.set_yticklabels([tp for tp, _ in cp], fontsize=8.5)
ax.set_xlim(0, 52); ax.set_xlabel("Số bài báo"); grid(ax, "x")
NP, NC = sum(v["papers"] for v in stats["corpus"].values()), sum(v["chunks"] for v in stats["corpus"].values())
ax.set_title(f"Corpus: {NP} bài báo, {NC:,} chunk theo chủ đề".replace(",", "."), loc="left")
save(fig, "fig9_corpus.png")

# Fig 10 - risk–coverage: coverage = tỉ lệ câu có đáp án được trả lời; risk = tỉ lệ câu trả lời sai hoặc có claim
# không được hỗ trợ (trong số câu được trả lời). So hai hệ ở hai điều kiện top-k.
def risk(o):
    c = o["counts"]
    return (c["correct_unsupported"] + c["wrong"]) / o["n_answered"] if o["n_answered"] else math.nan


curve = pd.read_csv(THR_CURVE) if THR_CURVE.exists() else None
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
for ax, qt in zip(axes, ("single", "multihop")):
    if curve is not None:  # RAG thuần + refuse khi cosine của đoạn đầu < τ, quét τ
        for r, ls in (("k4", "-"), ("k8", "--")):
            c = curve[(curve.run == r) & (curve.type == qt)].sort_values("coverage")
            ax.plot(c.coverage * 100, c.risk * 100, color=C["lc"], lw=1.1, ls=ls, alpha=0.75, zorder=1)
    for s in SYS:
        pts = [(stats["runs"][r]["types"][qt]["outcomes"][s], r) for r in ("k4", "k8")
               if r in stats["runs"] and qt in stats["runs"][r]["types"]]
        xs = [o["coverage"] * 100 for o, _ in pts]
        ys = [risk(o) * 100 for o, _ in pts]
        for (o, r), xx, yy in zip(pts, xs, ys):
            ax.scatter(xx, yy, s=70, color=C[s], marker="o" if r == "k4" else "s", zorder=2, edgecolor=SURF, lw=1.2)
    ax.set_title(TYPE_VI[qt], loc="left")
    ax.set_xlim(40, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Coverage: % câu có đáp án được trả lời"); grid(ax, "y"); grid(ax, "x")
axes[0].set_ylabel("Risk: % câu trả lời sai\nhoặc có claim không được hỗ trợ")
axes[1].legend(handles=[*[plt.Line2D([], [], color=C[s], marker="o", ls="", label=NAMES[s]) for s in SYS],
                        *[plt.Line2D([], [], color=INK2, marker=mk, ls="", label=f"top-k = {stats['runs'][r]['top_k']}")
                          for r, mk in (("k4", "o"), ("k8", "s")) if r in stats["runs"]],
                        plt.Line2D([], [], color=C["lc"], lw=1.1, label="RAG thuần + ngưỡng (top-4)"),
                        plt.Line2D([], [], color=C["lc"], lw=1.1, ls="--", label="RAG thuần + ngưỡng (top-8)")],
               frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout()
save(fig, "fig10_risk_coverage.png")

print(json.dumps({r: {qt: {s: {m: round(v[0], 3) for m, v in d["main"][s].items()} for s in SYS}
                      for qt, d in stats["runs"][r]["types"].items()} for r in stats["runs"]}, indent=1))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
