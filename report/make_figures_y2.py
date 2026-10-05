"""Tính số liệu + vẽ biểu đồ cho báo cáo nội dung 2 -> report/figures/y2_*.png, report/stats_y2.json

Đọc results/kg30 (GraphRAG vs Vector RAG, 30 bài) và data/kg/k30/graph.json (KG).
    python report/make_figures_y2.py
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from evaluate import load_jsonl, load_testset  # noqa: E402
from evaluate_graphrag import TESTSET_FILES, gold_sources  # noqa: E402
from kg_build import load_papers  # noqa: E402

RUN = config.RESULTS_DIR / "kg30"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# palette giống báo cáo nội dung 1 (categorical slot 1-2, light mode)
C = {"vec": "#2a78d6", "gr": "#eb6834"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
NAMES = {"vec": "Vector RAG", "gr": "GraphRAG"}
SYS = {"vec": "vector_rag", "gr": "graphrag"}
TYPES = ["all", "single", "multihop"]
TYPE_VI = {"all": "Tất cả câu có đáp án", "single": "Single-hop", "multihop": "Multi-hop",
           "unanswerable": "Ngoài corpus"}
# (khoá trong scores, tên hiển thị, chiều tốt, là tỉ lệ %)
METRICS = [("context_precision", "Retrieval precision", "↑", False),
           ("faithfulness", "Faithfulness", "↑", False),
           ("answer_relevancy", "Answer relevancy", "↑", False),
           ("hallucination", "Hallucination rate", "↓", True),
           ("answer_correctness", "Answer correctness", "↑", True)]
EXTRA = ["context_ap", "source_precision", "retrieval_hit", "gold_chunk_recall", "context_recall",
         "kg_relation_precision", "faithfulness_text", "hallucination_text", "refusal"]

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


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def boot(x, n=2000):
    x = np.asarray([v for v in x if not isnan(v)], float)
    if len(x) == 0:
        return [math.nan] * 3
    means = rng.choice(x, size=(n, len(x))).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(x.mean()), float(lo), float(hi)]


# ---------------- load ----------------
PAPERS = load_papers(config.KG_K)
TS = {t["id"]: t for t in load_testset(TESTSET_FILES)
      if t["type"] == "unanswerable" or set(gold_sources(t)) <= set(PAPERS)}
A = {s: load_jsonl(RUN / f"answers_{SYS[s]}.jsonl") for s in SYS}
S = {s: load_jsonl(RUN / f"scores_{SYS[s]}.jsonl") for s in SYS}
INFO = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
G = json.loads((config.KG_DIR / f"k{config.KG_K}" / "graph.json").read_text(encoding="utf-8"))
PINFO = json.loads(config.KG_PAPERS_FILE.read_text(encoding="utf-8"))["papers"]


def ids_of(qt):
    ids = [i for i in TS if i in S["vec"] and i in S["gr"]]
    if qt == "all":
        return [i for i in ids if TS[i]["type"] != "unanswerable"]
    return [i for i in ids if TS[i]["type"] == qt]


# ---------------- stats ----------------
stats = {"run": "kg30", "environment": INFO["environment"], "top_k": INFO["top_k"], "graphrag": INFO["graphrag"],
         "kg": G["stats"], "types": {}}
for qt in [*TYPES, "unanswerable"]:
    ids = ids_of(qt)
    ts = stats["types"][qt] = {"n": len(ids), "main": {}, "extra": {}, "paired": {}}
    for s in SYS:
        ts["main"][s] = {m: boot([S[s][i].get(m) for i in ids]) for m, *_ in METRICS}
        ts["extra"][s] = {k: (float(np.nanmean([S[s][i].get(k, math.nan) for i in ids]))
                              if any(not isnan(S[s][i].get(k)) for i in ids) else None) for k in EXTRA}
        ts["extra"][s]["latency_s"] = float(np.mean([A[s][i]["latency_s"] for i in ids]))
    for m, *_ in METRICS:  # chênh lệch theo cặp GraphRAG − Vector RAG trên cùng câu hỏi (như nội dung 1)
        d = np.array([S["gr"][i][m] - S["vec"][i][m] for i in ids
                      if not (isnan(S["gr"][i].get(m)) or isnan(S["vec"][i].get(m)))], float)
        if len(d) == 0 or not d.any() and qt == "unanswerable":
            ts["paired"][m] = None
            continue
        bs = rng.choice(d, size=(5000, len(d))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ts["paired"][m] = {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d),
                           "p": float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))}

# Entity "quá chung" (gắn với >= 50 chunk) trong số entity hạt giống -> tỉ lệ lấy được đoạn gốc
cnt = {e["name"]: len(e["chunk_ids"]) for e in G["entities"]}
hub = {True: [], False: []}
for i in ids_of("all"):
    has_hub = max(cnt.get(n, 0) for n in A["gr"][i]["kg_entities"]) >= 50
    hub[has_hub].append(S["gr"][i]["gold_chunk_recall"])
stats["hub"] = {"threshold": 50, "with": {"n": len(hub[True]), "gold_chunk_recall": float(np.mean(hub[True]))},
                "without": {"n": len(hub[False]), "gold_chunk_recall": float(np.mean(hub[False]))}}
stats["chunks_per_entity"] = {"median": float(np.median(list(cnt.values()))),
                              "p99": float(np.percentile(list(cnt.values()), 99)),
                              "max": max(cnt.values())}
stats["hub_examples"] = [{"name": e["name"], "chunks": len(e["chunk_ids"])}
                         for e in sorted(G["entities"], key=lambda e: -len(e["chunk_ids"]))[:5]]
# Liên kết giữa 2 nhánh chủ đề: entity xuất hiện ở bài của cả Quantum Security và Quantum ML
ptopic = {p["source"]: p["topic"] for p in PINFO}
ent_topics = [{ptopic[s] for s in e["sources"]} for e in G["entities"]]
stats["cross_topic"] = {"entities_both_topics": sum(len(t) == 2 for t in ent_topics),
                        "relations_both_topics": sum(len({ptopic[s] for s in r["sources"]}) == 2
                                                     for r in G["relations"])}
top = sorted(G["entities"], key=lambda e: (-len(e["sources"]), -len(e["chunk_ids"])))[:10]
stats["top_entities"] = [{"name": e["name"], "papers": len(e["sources"]), "chunks": len(e["chunk_ids"])} for e in top]
stats["papers_per_entity"] = dict(Counter(min(len(e["sources"]), 5) for e in G["entities"]))
stats["corpus"] = dict(Counter(p["topic"] for p in PINFO))
stats["corpus_added"] = dict(Counter(p["topic"] for p in PINFO if p.get("added_for_y2")))
stats["corpus_chunks"] = {t: sum(p["chunks"] for p in PINFO if p["topic"] == t) for t in stats["corpus"]}
raw_mh = json.loads((ROOT / "testset" / "multihop_kg_raw.json").read_text(encoding="utf-8"))
raw_sh = json.loads((ROOT / "testset" / "singlehop_kg_raw.json").read_text(encoding="utf-8"))
stats["testset"] = {
    "single_reused": sum(TS[i]["type"] == "single" and not i.startswith("s") for i in TS),
    "single_new": sum(i.startswith("s") for i in TS), "single_new_generated": len(raw_sh),
    "multihop_reused": sum(TS[i]["type"] == "multihop" and not i.startswith("g") for i in TS),
    "multihop_new": sum(i.startswith("g") for i in TS),
    "multihop_new_generated": sum(set(t["sources"]) <= set(PAPERS) for t in raw_mh),
    "unanswerable": sum(TS[i]["type"] == "unanswerable" for i in TS)}
stats["testset"]["single"] = stats["testset"]["single_reused"] + stats["testset"]["single_new"]
(ROOT / "report" / "stats_y2.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


# ================= FIGURES =================
def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))


def chain(ax, y, labels, color, fill, title):
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.4); ax.axis("off")
    ax.set_title(title, loc="left")
    n = len(labels)
    w = 9.6 / n
    xs = [0.2 + w * (k + 0.5) for k in range(n)]
    for k, (x, t) in enumerate(zip(xs, labels)):
        plain = k in (0, n - 1)
        box(ax, x, y, w - 0.35, 1.0, t, fc="#ffffff" if plain else fill, ec=color if not plain else MUTED, fs=8)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + (w - 0.35) / 2, y, b - (w - 0.35) / 2, y)


# Fig 1 - dựng KG + hai hệ
fig, axes = plt.subplots(3, 1, figsize=(9.5, 5.6))
chain(axes[0], 0.7, ["30 bài\n2.765 chunk", "LLM trích entity\n+ quan hệ\n(qwen2.5:7b)", "Gộp entity\n(viết tắt, số nhiều)",
                     "Embed entity\n+ quan hệ\n(nomic)", "Knowledge\nGraph"],
      INK2, "#efeee9", "(a) Dựng Knowledge Graph (một lần, kg_build.py)")
chain(axes[1], 0.7, ["Câu hỏi", "Chroma top-4\n(giới hạn 30 bài)", "Generate\n(qwen2.5:7b)", "Câu trả lời"],
      C["vec"], "#eaf2fc", "(b) Vector RAG: truy xuất theo embedding đoạn văn")
chain(axes[2], 0.7, ["Câu hỏi", "Khớp entity\n(embedding\n+ tên)", "Mở rộng\nquan hệ 1 bước", "Chọn 4 chunk\ngắn với entity\n/ quan hệ",
                     "Generate\n(mô tả KG\n+ 4 đoạn văn)", "Câu trả lời"],
      C["gr"], "#fdeee7", "(c) GraphRAG: truy xuất qua Knowledge Graph (local search)")
fig.tight_layout()
save(fig, "y2_fig1_pipeline.png")

# Fig 2 - KG
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [1.25, 1]})
types = sorted(G["stats"]["entity_types"].items(), key=lambda kv: kv[1])
a1.barh([t for t, _ in types], [v for _, v in types], color=INK2, height=0.62, edgecolor=SURF, lw=2)
for y, (_, v) in enumerate(types):
    a1.text(v + 30, y, f"{v:,}".replace(",", "."), va="center", fontsize=7.5, color=INK2)
a1.set_title(f"(a) {G['stats']['entities']:,} entity theo loại".replace(",", "."), loc="left")
a1.set_xlim(0, max(v for _, v in types) * 1.18); grid(a1, "x"); a1.tick_params(labelsize=8)
ppe = stats["papers_per_entity"]
ks = [1, 2, 3, 4, 5]
vals = [ppe.get(k, 0) for k in ks]
a2.bar(range(5), vals, color=INK2, width=0.62, edgecolor=SURF, lw=2)
for x, v in enumerate(vals):
    a2.text(x, v * 1.15, f"{v:,}".replace(",", "."), ha="center", fontsize=7.5, color=INK2)
a2.set_yscale("log"); a2.set_ylim(1, max(vals) * 4)
a2.set_xticks(range(5)); a2.set_xticklabels(["1", "2", "3", "4", "≥ 5"])
a2.set_xlabel("Số bài báo chứa entity"); a2.set_title("(b) Entity xuất hiện ở bao nhiêu bài (thang log)", loc="left")
grid(a2)
fig.tight_layout()
save(fig, "y2_fig2_kg.png")


# Fig 3 - năm chỉ số chính theo loại câu hỏi
def bar_ci(ax, x, v, color, pct, w=0.36, hatch=None, label=None):
    mean, lo, hi = v
    if isnan(mean):
        return
    f = 100 if pct else 1
    ax.bar(x, mean * f, w, color=color, edgecolor=SURF, lw=2, hatch=hatch, label=label)
    ax.errorbar(x, mean * f, yerr=[[(mean - lo) * f], [(hi - mean) * f]], color=INK, capsize=3, lw=1)
    ax.text(x, hi * f + (1.5 if pct else 0.015), f"{mean * f:.0f}%" if pct else f"{mean:.2f}",
            ha="center", va="bottom", fontsize=7, color=INK2)


fig, axes = plt.subplots(1, 5, figsize=(13, 3.4))
for ax, (m, name, arrow_, pct) in zip(axes, METRICS):
    for k, qt in enumerate(TYPES):
        for j, s in enumerate(SYS):
            bar_ci(ax, k + (j - 0.5) * 0.38, stats["types"][qt]["main"][s][m], C[s], pct,
                   hatch="///" if s == "gr" else None, label=NAMES[s] if k == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Tất cả", "Single", "Multi"], fontsize=8.5)
    ax.set_ylim(0, 115 if pct else 1.15); ax.set_title(f"{name} {arrow_}", fontsize=9.5)
    grid(ax)
axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
fig.tight_layout()
save(fig, "y2_fig3_main.png")

# Fig 4 - chênh lệch theo cặp (forest): (a) chỉ số thang 0–1, (b) chỉ số tỉ lệ (điểm %)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, want_pct, title in ((axes[0], False, "(a) Chỉ số thang 0–1"), (axes[1], True, "(b) Chỉ số tỉ lệ (điểm %)")):
    rows = [(qt, m, name) for m, name, _, pct in METRICS if pct == want_pct for qt in TYPES]
    f = 100 if want_pct else 1
    for y, (qt, m, name) in enumerate(reversed(rows)):
        d = stats["types"][qt]["paired"][m]
        sig = d["lo"] > 0 or d["hi"] < 0
        col = C["gr"] if sig else MUTED
        ax.plot([d["lo"] * f, d["hi"] * f], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(d["delta"] * f, y, "o", color=col, ms=6, mec=SURF, mew=1.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{name} · {'Tất cả' if qt == 'all' else TYPE_VI[qt]}" for qt, m, name in reversed(rows)],
                       fontsize=8)
    ax.set_xlabel("GraphRAG − Vector RAG" + (" (điểm %)" if want_pct else ""))
    ax.set_title(title, loc="left")
    grid(ax, "x")
fig.suptitle("Chênh lệch theo cặp trên cùng câu hỏi, khoảng tin cậy 95% (cam: không chứa 0)",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "y2_fig4_paired.png")

# Fig 5 - chẩn đoán truy xuất (câu có đáp án)
diag = [("gold_chunk_recall", "Lấy được\nđoạn gốc"), ("retrieval_hit", "Có đủ\nbài nguồn"),
        ("source_precision", "Đoạn thuộc\nbài nguồn"), ("context_recall", "Context\nrecall")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3))
for ax, qt in ((a1, "single"), (a2, "multihop")):
    for k, (key, lab) in enumerate(diag):
        for j, s in enumerate(SYS):
            v = stats["types"][qt]["extra"][s][key]
            x = k + (j - 0.5) * 0.38
            ax.bar(x, v * 100, 0.36, color=C[s], edgecolor=SURF, lw=2, hatch="///" if s == "gr" else None,
                   label=NAMES[s] if k == 0 else None)
            ax.text(x, v * 100 + 1.5, f"{v * 100:.0f}%", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(diag))); ax.set_xticklabels([l for _, l in diag], fontsize=8)
    ax.set_ylim(0, 110); ax.set_title(f"{TYPE_VI[qt]} (n = {stats['types'][qt]['n']})", loc="left"); grid(ax)
a1.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
save(fig, "y2_fig5_retrieval.png")
print("saved stats_y2.json + 5 figures")
