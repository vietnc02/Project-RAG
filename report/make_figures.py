"""Tính số liệu + vẽ biểu đồ cho báo cáo ý 1 -> report/figures/*.png, report/stats.json"""
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
from common import is_refusal  # noqa: E402
from evaluate import load_jsonl  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (dataviz reference palette, light mode) ----
C_LC, C_SR = "#2a78d6", "#eb6834"          # categorical slot 1, 2
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
SURF = "#fcfcfb"
NAMES = {"lc": "LangChain RAG", "sr": "LangGraph Self-RAG"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def grid_x(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------- load data ----------------
R = config.RESULTS_DIR
ts_all = json.load(open(config.TESTSET_FILE, encoding="utf-8"))
ts = {t["id"]: t for t in ts_all if t["keep"]}
A = {"lc": load_jsonl(R / "answers_langchain_rag.jsonl"), "sr": load_jsonl(R / "answers_langgraph_selfrag.jsonl")}
S = {"lc": load_jsonl(R / "scores_langchain_rag.jsonl"), "sr": load_jsonl(R / "scores_langgraph_selfrag.jsonl")}
ids = sorted(ts)
rng = np.random.default_rng(config.SEED)


def vals(sys_, m, skip_nan=True):
    v = [S[sys_][i][m] for i in ids]
    return [x for x in v if not (skip_nan and isinstance(x, float) and math.isnan(x))]


def boot(x, n=2000):
    x = np.asarray(x, float)
    means = rng.choice(x, size=(n, len(x))).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


stats = {"n": len(ids), "main": {}, "extra": {}, "paired": {}}
for s in ("lc", "sr"):
    stats["main"][s] = {m: boot(vals(s, m)) for m in ("faithfulness", "answer_relevancy", "hallucination")}
    nc = sum(S[s][i]["n_claims"] for i in ids)
    nu = sum(S[s][i]["n_unsupported"] for i in ids)
    answered = [i for i in ids if not S[s][i]["refusal"]]
    stats["extra"][s] = {
        "claim_halluc_rate": nu / nc, "n_claims": nc, "n_unsupported": nu,
        "halluc_among_answered": sum(S[s][i]["hallucination"] for i in answered) / len(answered),
        "n_answered": len(answered),
        "answer_correctness": float(np.mean(vals(s, "answer_correctness"))),
        "refusal_rate": float(np.mean(vals(s, "refusal"))),
        "retrieval_hit": float(np.mean(vals(s, "retrieval_hit"))),
        "latency_s": float(np.mean([A[s][i]["latency_s"] for i in ids])),
        "llm_calls": float(np.mean([A[s][i]["llm_calls"] for i in ids])),
    }

# paired differences (Self-RAG - LC)
for m in ("faithfulness", "answer_relevancy", "hallucination"):
    d = []
    for i in ids:
        a, b = S["lc"][i][m], S["sr"][i][m]
        if any(isinstance(x, float) and math.isnan(x) for x in (a, b)):
            continue
        d.append(b - a)
    d = np.array(d, float)
    bs = rng.choice(d, size=(5000, len(d))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    stats["paired"][m] = {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d),
                          "p": float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))}
h = {k: sum(1 for i in ids if (S["lc"][i]["hallucination"], S["sr"][i]["hallucination"]) == k)
     for k in [(1, 1), (1, 0), (0, 1), (0, 0)]}
b_, c_ = h[(1, 0)], h[(0, 1)]
stats["mcnemar"] = {"both": h[(1, 1)], "only_lc": b_, "only_sr": c_, "neither": h[(0, 0)],
                    "p": min(1.0, 2 * sum(comb(b_ + c_, k) for k in range(min(b_, c_) + 1)) / 2 ** (b_ + c_)),
                    "fixed_by_refusal": sum(1 for i in ids if S["lc"][i]["hallucination"] == 1
                                            and S["sr"][i]["hallucination"] == 0 and S["sr"][i]["refusal"] == 1)}

# self-rag behaviour
ends = Counter()
for i in ids:
    t = A["sr"][i]["trace"][-1]
    ends["abstain" if t == "abstain" else t.split("(")[-1].rstrip(")")] += 1
stats["sr_behaviour"] = {"ends": dict(ends),
                         "rewrites": sum(A["sr"][i].get("rewrites", 0) > 0 for i in ids),
                         "regenerations": sum(A["sr"][i].get("regenerations", 0) > 0 for i in ids)}

# refusal breakdown (Self-RAG)
ref = [i for i in ids if is_refusal(A["sr"][i]["answer"])]
cat = {"lc_also_refused": [], "lc_hallucinated_or_wrong": [], "false_abstain": []}
for i in ref:
    if S["lc"][i]["refusal"]:
        cat["lc_also_refused"].append(i)
    elif S["lc"][i]["answer_correctness"] and S["lc"][i]["faithfulness"] >= 0.85:
        cat["false_abstain"].append(i)
    else:
        cat["lc_hallucinated_or_wrong"].append(i)
stats["sr_refusals"] = cat

# per topic
topic_ids = defaultdict(list)
for i in ids:
    topic_ids[ts[i]["topic"]].append(i)
stats["topics"] = {tp: {"n": len(L), **{s: sum(S[s][i]["hallucination"] for i in L) / len(L) for s in ("lc", "sr")}}
                   for tp, L in topic_ids.items()}

# corpus
papers_by_topic, chunks_by_topic = defaultdict(set), Counter()
with open(config.CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        papers_by_topic[c["topic"]].add(c["source"])
        chunks_by_topic[c["topic"]] += 1
stats["corpus"] = {tp: {"papers": len(p), "chunks": chunks_by_topic[tp]} for tp, p in papers_by_topic.items()}

# testset review
stats["review"] = {"generated": len(ts_all), "kept": sum(t["keep"] for t in ts_all),
                   "edited": sum(t.get("review_note") == "edited" for t in ts_all),
                   "dropped": sum(not t["keep"] for t in ts_all)}
stats["review"]["unchanged"] = stats["review"]["kept"] - stats["review"]["edited"]
(ROOT / "report" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


# ================= FIGURES =================
# Fig 1 - architecture of both systems
def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, label=None, color=INK2, rad=0.0, lx=0, ly=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7.5, color=INK2, ha="center", va="center",
                bbox=dict(fc=SURF, ec="none", pad=0.5))


fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), gridspec_kw={"height_ratios": [1, 2.3]})
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
ax.set_title("(a) LangChain RAG thuần - 1 lần gọi LLM", loc="left")
xs = [1.0, 3.4, 5.8, 8.4]
labels = ["Câu hỏi", "Retriever\n(Chroma, top-4)", "Generate\n(qwen2.5:7b)", "Câu trả lời"]
for x, t in zip(xs, labels):
    box(ax, x, 1.0, 1.8, 0.9, t, fc="#eaf2fc" if "Retr" in t or "Gen" in t else "#ffffff", ec=C_LC)
for a, b in zip(xs, xs[1:]):
    arrow(ax, a + 0.9, 1.0, b - 0.9, 1.0)

ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
ax.set_title("(b) LangGraph Self-RAG - vòng tự kiểm tra (trung bình 8.1 lần gọi LLM)", loc="left")
P = {"q": (0.8, 4.6), "ret": (2.8, 4.6), "grade": (5.2, 4.6), "gen": (7.6, 4.6),
     "gg": (7.6, 2.5), "end": (8.2, 0.6), "tq": (2.8, 2.5), "abs": (5.2, 0.6), "regen": (9.3, 3.55)}
fcs = "#fdeee7"
box(ax, *P["q"], 1.3, 0.8, "Câu hỏi")
box(ax, *P["ret"], 1.7, 0.8, "retrieve\n(top-4)", fc=fcs, ec=C_SR)
box(ax, *P["grade"], 2.0, 0.8, "grade_documents\n(chấm liên quan)", fc=fcs, ec=C_SR)
box(ax, *P["gen"], 1.5, 0.8, "generate", fc=fcs, ec=C_SR)
box(ax, *P["gg"], 2.3, 0.9, "grade_generation\n1) grounded? 2) useful?", fc=fcs, ec=C_SR)
box(ax, *P["tq"], 1.9, 0.8, "transform_query\n(viết lại truy vấn)", fc=fcs, ec=C_SR)
box(ax, *P["regen"], 1.3, 0.8, "regenerate\n(chặt hơn)", fc=fcs, ec=C_SR)
box(ax, *P["abs"], 1.9, 0.8, "abstain\n(từ chối trả lời)")
box(ax, *P["end"], 1.6, 0.8, "Câu trả lời")
arrow(ax, 1.45, 4.6, 1.95, 4.6)
arrow(ax, 3.65, 4.6, 4.2, 4.6)
arrow(ax, 6.2, 4.6, 6.85, 4.6, "có tài liệu\nliên quan", ly=0.68)
arrow(ax, 7.6, 4.2, 7.6, 2.95)
arrow(ax, 8.2, 2.05, 8.2, 1.0, "grounded\n& useful", lx=0.55)
arrow(ax, 8.75, 2.75, 9.1, 3.15, "không grounded\n(còn lượt)", lx=0.55, ly=-0.55)
arrow(ax, 9.1, 3.95, 8.35, 4.45, rad=0.25)
arrow(ax, 6.45, 2.5, 3.75, 2.5, "không useful (còn lượt)", lx=-0.55, ly=0.22)
arrow(ax, 4.6, 4.2, 3.4, 2.9, "0 tài liệu\n(còn lượt)", lx=-0.45, ly=0.1)
arrow(ax, 2.8, 2.9, 2.8, 4.2)
arrow(ax, 5.35, 4.2, 5.35, 1.0, "0 tài liệu\n(hết lượt)", lx=0.0, ly=0.55)
arrow(ax, 7.0, 2.05, 6.15, 0.8, "không grounded\n(hết lượt)", lx=-0.15, ly=-0.05)
fig.tight_layout()
save(fig, "fig1_architecture.png")

# Fig 2 - testset review
rv = stats["review"]
fig, ax = plt.subplots(figsize=(7, 1.9))
parts = [("Giữ nguyên", rv["unchanged"], C_LC), ("Đã sửa", rv["edited"], "#86b6ef"), ("Loại", rv["dropped"], "#c3c2b7")]
left = 0
for name, v, col in parts:
    ax.barh(0, v, left=left, color=col, height=0.5, edgecolor=SURF, lw=2)
    ax.text(left + v / 2, 0, f"{name}\n{v}", ha="center", va="center", fontsize=9,
            color="#ffffff" if col == C_LC else INK)
    left += v
ax.set_xlim(0, rv["generated"]); ax.set_yticks([])
ax.set_xlabel(f"Số câu hỏi (sinh tự động {rv['generated']} câu -> dùng {rv['kept']} câu sau duyệt tay)")
ax.spines["left"].set_visible(False)
save(fig, "fig2_testset_review.png")

# Fig 3 - main metrics small multiples with CI
mets = [("faithfulness", "Faithfulness (↑ tốt hơn)", False),
        ("answer_relevancy", "Answer relevancy (↑ tốt hơn)", False),
        ("hallucination", "Hallucination rate (↓ tốt hơn)", True)]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (m, title, pct) in zip(axes, mets):
    for k, s in enumerate(("lc", "sr")):
        mean, lo, hi = stats["main"][s][m]
        f = 100 if pct else 1
        ax.bar(k, mean * f, width=0.6, color=C_LC if s == "lc" else C_SR, edgecolor=SURF, lw=2)
        ax.errorbar(k, mean * f, yerr=[[(mean - lo) * f], [(hi - mean) * f]], color=INK, capsize=4, lw=1.1)
        ax.text(k, hi * f + (2 if pct else 0.02), f"{mean * f:.1f}%" if pct else f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LangChain\nRAG", "Self-RAG"])
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 55 if pct else 1.12)
    if pct:
        ax.set_ylabel("% câu hỏi")
    grid_y(ax)
fig.suptitle("Ba chỉ số chính (n = 86, thanh lỗi = khoảng tin cậy 95% bootstrap)", fontsize=11, color=INK, x=0.01,
             ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "fig3_main_metrics.png")

# Fig 4 - paired differences (forest plot)
fig, ax = plt.subplots(figsize=(8, 2.6))
rows = [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer relevancy"), ("hallucination", "Hallucination rate")]
for y, (m, name) in enumerate(reversed(rows)):
    p = stats["paired"][m]
    sig = not (p["lo"] <= 0 <= p["hi"])
    col = C_SR if sig else MUTED
    ax.plot([p["lo"], p["hi"]], [y, y], color=col, lw=2)
    ax.plot(p["delta"], y, "o", color=col, ms=8, mec=SURF, mew=1.5)
    ax.text(0.205, y, f"Δ = {p['delta']:+.3f}  [{p['lo']:+.3f}, {p['hi']:+.3f}]  p ≈ {p['p']:.2f}",
            va="center", fontsize=8.5, color=INK2)
ax.axvline(0, color=INK2, lw=1, ls="--")
ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n in reversed(rows)])
ax.set_xlim(-0.2, 0.2)
ax.set_xlabel("Chênh lệch theo cặp: Self-RAG − LangChain RAG (cùng câu hỏi)")
ax.set_title("Chênh lệch có ý nghĩa thống kê khi khoảng tin cậy không cắt 0 (màu cam)", loc="left", fontsize=10)
grid_x(ax)
save(fig, "fig4_paired_diff.png")

# Fig 5 - secondary metrics + cost
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
sec = [("claim_halluc_rate", "Tỉ lệ claim\nsai"), ("halluc_among_answered", "Hallucination\n(chỉ câu có trả lời)"),
       ("refusal_rate", "Tỉ lệ\ntừ chối"), ("answer_correctness", "Answer\ncorrectness"), ("retrieval_hit", "Retrieval\nhit")]
x = np.arange(len(sec)); w = 0.36
for k, s in enumerate(("lc", "sr")):
    v = [stats["extra"][s][m] * 100 for m, _ in sec]
    ax.bar(x + (k - 0.5) * w, v, w, color=C_LC if s == "lc" else C_SR, edgecolor=SURF, lw=2, label=NAMES[s])
    for xi, vi in zip(x + (k - 0.5) * w, v):
        ax.text(xi, vi + 1.5, f"{vi:.1f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in sec], fontsize=8.5)
ax.set_ylabel("%"); ax.set_ylim(0, 110); grid_y(ax)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)
ax.set_title("Chỉ số phụ", loc="left")
ax = axes[1]
for k, s in enumerate(("lc", "sr")):
    v = stats["extra"][s]["latency_s"]
    ax.bar(k, v, 0.6, color=C_LC if s == "lc" else C_SR, edgecolor=SURF, lw=2)
    ax.text(k, v + 0.6, f"{v:.1f} s\n({stats['extra'][s]['llm_calls']:.1f} lần gọi LLM)", ha="center", fontsize=8, color=INK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["LangChain\nRAG", "Self-RAG"]); ax.set_ylim(0, 40)
ax.set_ylabel("giây / câu"); grid_y(ax)
ax.set_title("Chi phí", loc="left")
fig.tight_layout()
save(fig, "fig5_secondary.png")

# Fig 6 - hallucination transitions
mc = stats["mcnemar"]
fig, ax = plt.subplots(figsize=(8, 2.6))
cats = [("Không hệ nào hallucination", mc["neither"], "#c3c2b7"),
        ("Cả hai hệ đều hallucination", mc["both"], INK2),
        ("Chỉ LangChain (Self-RAG đã sửa được)", mc["only_lc"], C_LC),
        ("Chỉ Self-RAG (phát sinh mới)", mc["only_sr"], C_SR)]
for y, (name, v, col) in enumerate(reversed(cats)):
    ax.barh(y, v, color=col, height=0.6, edgecolor=SURF, lw=2)
    ax.text(v + 0.8, y, str(v), va="center", fontsize=9, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([c[0] for c in reversed(cats)])
ax.set_xlabel("Số câu hỏi (n = 86)"); grid_x(ax); ax.set_xlim(0, 60)
ax.set_title(f"So khớp từng câu: hallucination của hai hệ (McNemar p ≈ {mc['p']:.2f})", loc="left", fontsize=10)
save(fig, "fig6_halluc_transitions.png")

# Fig 7 - Self-RAG behaviour
e = stats["sr_behaviour"]["ends"]
lab = {"useful": "Qua kiểm tra (grounded & useful)", "abstain": "Abstain (từ chối)",
       "not_useful": "Hết lượt, chưa useful", "refusal": "Model tự trả lời 'không tìm thấy'"}
items = sorted(e.items(), key=lambda kv: kv[1])
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for y, (k, v) in enumerate(items):
    ax.barh(y, v, color=C_SR if k == "useful" else "#f3b79e", height=0.6, edgecolor=SURF, lw=2)
    ax.text(v + 0.8, y, str(v), va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([lab.get(k, k) for k, _ in items], fontsize=8.5)
ax.set_xlim(0, 90); grid_x(ax); ax.set_title("Trạng thái kết thúc của Self-RAG", loc="left", fontsize=10)
ax = axes[1]
sb = stats["sr_behaviour"]
for y, (name, v) in enumerate([("Có viết lại truy vấn", sb["rewrites"]), ("Có sinh lại (not grounded)", sb["regenerations"])]):
    ax.barh(y, v, color=C_SR, height=0.6, edgecolor=SURF, lw=2)
    ax.text(v + 0.3, y, str(v), va="center", fontsize=9, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Có viết lại truy vấn", "Có sinh lại\n(not grounded)"], fontsize=8.5)
ax.set_xlim(0, 15); grid_x(ax); ax.set_title("Số câu kích hoạt vòng lặp", loc="left", fontsize=10)
fig.tight_layout()
save(fig, "fig7_selfrag_behaviour.png")

# Fig 8 - refusal analysis
rc = stats["sr_refusals"]
fig, ax = plt.subplots(figsize=(8, 2.2))
rr = [("Đúng: LangChain cũng không tìm thấy", len(rc["lc_also_refused"]), "#c3c2b7"),
      ("Đúng: LangChain trả lời sai / bịa", len(rc["lc_hallucinated_or_wrong"]), C_LC),
      ("Sai: từ chối oan (tài liệu có đáp án)", len(rc["false_abstain"]), C_SR)]
for y, (name, v, col) in enumerate(reversed(rr)):
    ax.barh(y, v, color=col, height=0.6, edgecolor=SURF, lw=2)
    ax.text(v + 0.15, y, str(v), va="center", fontsize=9, color=INK)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in reversed(rr)])
ax.set_xlim(0, 8); ax.set_xlabel("Số câu"); grid_x(ax)
ax.set_title(f"Phân loại {len(ref)} câu Self-RAG từ chối trả lời", loc="left", fontsize=10)
save(fig, "fig8_refusals.png")

# Fig 9 - per topic hallucination
tps = sorted(stats["topics"].items(), key=lambda kv: kv[1]["n"])
fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(tps)); hgt = 0.36
for k, s in enumerate(("lc", "sr")):
    v = [d[s] * 100 for _, d in tps]
    ax.barh(y + (0.5 - k) * hgt, v, hgt, color=C_LC if s == "lc" else C_SR, edgecolor=SURF, lw=1.5, label=NAMES[s])
    for yi, vi in zip(y + (0.5 - k) * hgt, v):
        ax.text(vi + 1, yi, f"{vi:.0f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([f"{tp} (n={d['n']})" for tp, d in tps], fontsize=8.5)
ax.set_xlabel("Hallucination rate (%)"); ax.set_xlim(0, 110); grid_x(ax)
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.set_title("Hallucination rate theo chủ đề (chủ đề có n nhỏ chỉ mang tính tham khảo)", loc="left", fontsize=10)
save(fig, "fig9_topics.png")

# Fig 10 - corpus
cp = sorted(stats["corpus"].items(), key=lambda kv: kv[1]["papers"])
fig, ax = plt.subplots(figsize=(8, 3.2))
for yy, (tp, d) in enumerate(cp):
    ax.barh(yy, d["papers"], color=C_LC, height=0.6, edgecolor=SURF, lw=2)
    ax.text(d["papers"] + 0.4, yy, f"{d['papers']} bài · {d['chunks']} chunk", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(cp))); ax.set_yticklabels([tp for tp, _ in cp], fontsize=8.5)
ax.set_xlim(0, 52); ax.set_xlabel("Số bài báo"); grid_x(ax)
ax.set_title("Corpus: 95 bài báo, 6.649 chunk theo chủ đề", loc="left", fontsize=10)
save(fig, "fig10_corpus.png")

print(json.dumps({k: stats[k] for k in ("main", "paired", "mcnemar")}, indent=1)[:1500])
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
