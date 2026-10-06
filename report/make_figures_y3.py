"""Tính số liệu + vẽ biểu đồ cho báo cáo nội dung 3 -> report/figures/y3_*.png, report/stats_y3.json

Đọc results/kg_scale (GraphRAG với KG k = 10 / 20 / 30 bài; chất lượng KG từ kg_quality.py).
    python report/make_figures_y3.py
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
import config  # noqa: E402
from evaluate import load_jsonl  # noqa: E402
from evaluate_kg_scale import KS, OUT as RUN, load_items  # noqa: E402

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# k là thứ tự có độ lớn -> thang tuần tự một màu (xanh, bậc 250 / 400 / 600 của bảng màu tham chiếu, ordinal)
CK = {10: "#86b6ef", 20: "#3987e5", 30: "#184f95"}
# tập câu hỏi là nhóm phân loại -> slot 1, 2, 3 (thứ tự cố định, như báo cáo nội dung 1-2)
CS = {"in_scope": "#2a78d6", "S10": "#eb6834", "all": "#1baf7a"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
SET_VI = {"in_scope": "Trong phạm vi KG", "S10": "S10 (cố định, 15 câu)", "S20": "S20 (cố định, 37 câu)",
          "all": "Toàn bộ 62 câu", "out_scope": "Ngoài phạm vi KG", "unanswerable": "Ngoài corpus"}
# (khoá trong scores, tên hiển thị, là tỉ lệ %)
METRICS = [("context_precision", "Retrieval precision ↑", False), ("faithfulness", "Faithfulness ↑", False),
           ("answer_relevancy", "Answer relevancy ↑", False), ("hallucination", "Hallucination rate ↓", True),
           ("answer_correctness", "Answer correctness ↑", True)]
EXTRA = ["gold_chunk_recall", "retrieval_hit", "source_precision", "context_recall", "kg_relation_precision",
         "faithfulness_text", "hallucination_text", "refusal"]

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


def paired(a, b):
    """Hiệu theo cặp (a − b) trên cùng câu hỏi, CI 95% bootstrap 5000 lần + p hai phía (như nội dung 1-2)."""
    d = np.array([x - y for x, y in zip(a, b) if not (isnan(x) or isnan(y))], float)
    if len(d) == 0:
        return None
    bs = rng.choice(d, size=(5000, len(d))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 1.0 if not d.any() else float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d), "p": p}


# ---------------- load ----------------
ITEMS = {t["id"]: t for t in load_items()}
SC = {k: load_jsonl(RUN / f"k{k}" / "scores_graphrag.jsonl") for k in KS}
AN = {k: load_jsonl(RUN / f"k{k}" / "answers_graphrag.jsonl") for k in KS}
INFO = json.loads((RUN / "k10" / "run_info.json").read_text(encoding="utf-8"))
KQ = json.loads((RUN / "kg_quality.json").read_text(encoding="utf-8"))
COV = pd.DataFrame(KQ["coverage"])
GROWTH = pd.read_csv(RUN / "kg_growth.csv")
QCOV = pd.read_csv(RUN / "kg_coverage_per_question.csv")
missing = {k: len(ITEMS) - len(SC[k]) for k in KS if len(SC[k]) < len(ITEMS)}
if missing:
    raise SystemExit(f"Chưa chấm xong: {missing} câu (chạy python evaluate_kg_scale.py)")


def ids_of(name: str, k: int) -> list[str]:
    ans = [i for i, t in ITEMS.items() if t["type"] != "unanswerable"]
    return {"in_scope": [i for i in ans if ITEMS[i]["scope"] <= k],
            "in_single": [i for i in ans if ITEMS[i]["scope"] <= k and ITEMS[i]["type"] == "single"],
            "in_multihop": [i for i in ans if ITEMS[i]["scope"] <= k and ITEMS[i]["type"] == "multihop"],
            "S10": [i for i in ans if ITEMS[i]["scope"] == 10], "S20": [i for i in ans if ITEMS[i]["scope"] <= 20],
            "all": ans, "out_scope": [i for i in ans if ITEMS[i]["scope"] > k],
            "unanswerable": [i for i, t in ITEMS.items() if t["type"] == "unanswerable"]}[name]


# ---------------- stats ----------------
stats = {"environment": INFO["environment"], "top_k": INFO["top_k"], "graphrag": INFO["graphrag"],
         "kg": KQ["kg"], "growth": KQ["new_entities_per_paper_random_orders"],
         "projection_check": KQ["projection_check"], "gold": KQ["gold"], "relation_audit": KQ.get("relation_audit"),
         "coverage": KQ["coverage"], "sets": {}, "paired": {}}
for name in ("in_scope", "in_single", "in_multihop", "S10", "S20", "all", "out_scope", "unanswerable"):
    for k in KS:
        if name == "S20" and k < 20 or name == "out_scope" and k == 30:
            continue
        ids = ids_of(name, k)
        st = stats["sets"].setdefault(name, {})[str(k)] = {"n": len(ids)}
        st["main"] = {m: boot([SC[k][i].get(m) for i in ids]) for m, *_ in METRICS}
        st["extra"] = {e: (float(np.nanmean([SC[k][i].get(e, math.nan) for i in ids]))
                           if any(not isnan(SC[k][i].get(e)) for i in ids) else None) for e in EXTRA}
        st["extra"]["latency_s"] = float(np.mean([AN[k][i]["latency_s"] for i in ids]))
for name, a, b in (("S10", 10, 30), ("S10", 20, 30), ("S10", 10, 20), ("S20", 20, 30), ("all", 10, 30),
                   ("all", 20, 30), ("unanswerable", 10, 30)):
    ids = ids_of(name, a)
    stats["paired"][f"{name}:{a}-{b}"] = {m: paired([SC[a][i].get(m) for i in ids], [SC[b][i].get(m) for i in ids])
                                         for m, *_ in METRICS}

# Kết cục của 62 câu có đáp án theo k: trong / ngoài phạm vi KG x đúng / sai / từ chối
outcome = {}
for k in KS:
    o = {"in_correct": 0, "in_wrong": 0, "in_refused": 0, "out_refused": 0, "out_correct": 0, "out_wrong": 0}
    for i in ids_of("all", k):
        s, side = SC[k][i], "in" if ITEMS[i]["scope"] <= k else "out"
        o[f"{side}_{'refused' if s['refusal'] else 'correct' if s['answer_correctness'] else 'wrong'}"] += 1
    outcome[str(k)] = o
stats["outcome"] = outcome
# Câu ngoài phạm vi KG mà vẫn đúng: multi-hop có 1 bài nguồn trong KG? câu trả lời có claim ngoài context?
from evaluate_graphrag import gold_sources  # noqa: E402
from kg_build import load_papers  # noqa: E402
stats["out_correct"] = {}
for k in (10, 20):
    first = set(load_papers(k))
    ok = [i for i in ids_of("out_scope", k) if SC[k][i]["answer_correctness"] and not SC[k][i]["refusal"]]
    stats["out_correct"][str(k)] = {
        "n": len(ok),
        "multihop_one_source_in_kg": sum(ITEMS[i]["type"] == "multihop" and any(s in first for s in gold_sources(ITEMS[i]))
                                         for i in ok),
        "faithfulness_below_1": sum(SC[k][i]["faithfulness"] < 0.999 for i in ok)}

# Độ đúng của câu trả lời theo độ phủ thực thể của câu hỏi (gộp các câu trong phạm vi KG ở mọi k)
rows = []
for k in KS:
    for i in ids_of("in_scope", k):
        c = QCOV[(QCOV.k == k) & (QCOV.id == i)].iloc[0]
        rows.append({"k": k, "full": bool(c.entity_coverage >= 1), "rel": c.relation_completeness,
                     "correct": SC[k][i]["answer_correctness"], "cp": SC[k][i]["context_precision"]})
cva = pd.DataFrame(rows)
stats["coverage_vs_answer"] = {
    lab: {"n": int(len(g)), "answer_correctness": boot(g.correct.tolist()), "context_precision": boot(g.cp.tolist())}
    for lab, g in (("full", cva[cva.full]), ("partial", cva[~cva.full]))}
hi_rel = cva.rel >= 0.5
stats["relation_vs_answer"] = {
    lab: {"n": int(len(g)), "answer_correctness": boot(g.correct.tolist())}
    for lab, g in (("rel_ge_50", cva[hi_rel]), ("rel_lt_50", cva[~hi_rel & cva.rel.notna()]))}
stats["testset"] = {"scope": {str(s): sum(t["scope"] == s for t in ITEMS.values()) for s in (0, 10, 20, 30)},
                    "single": sum(t["type"] == "single" for t in ITEMS.values()),
                    "multihop": sum(t["type"] == "multihop" for t in ITEMS.values()),
                    "unanswerable": sum(t["type"] == "unanswerable" for t in ITEMS.values())}
papers = json.loads(config.KG_PAPERS_FILE.read_text(encoding="utf-8"))["papers"]
stats["papers_by_k"] = {str(k): {"Quantum Security": sum(p["topic"] == "Quantum Security" for p in papers[:k]),
                                 "Quantum Machine Learning": sum(p["topic"] != "Quantum Security" for p in papers[:k]),
                                 "added_for_y2": sum(p["added_for_y2"] for p in papers[:k])} for k in KS}
raw = json.loads((ROOT / "testset" / "kg_gold_raw.json").read_text(encoding="utf-8"))
gold = json.loads((ROOT / "testset" / "kg_gold.json").read_text(encoding="utf-8"))
stats["gold_size"] = {"raw_entities": sum(len(g["entities"]) for g in raw),
                      "raw_relations": sum(len(g["relations"]) for g in raw),
                      "entities": sum(len(g["entities"]) for g in gold),
                      "relations": sum(len(g["relations"]) for g in gold)}
stats["scope_by_type"] = {f"{tp}:{s}": sum(t["type"] == tp and t["scope"] == s for t in ITEMS.values())
                          for tp in ("single", "multihop") for s in (10, 20, 30)}
(ROOT / "report" / "stats_y3.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------- figures ----------------
def cov(set_name, k, col):
    r = COV[(COV["set"] == set_name) & (COV.k == k)].iloc[0]
    if col + "_ci95" in r:
        lo, hi = map(float, r[col + "_ci95"].strip("[]").split(","))
        return float(r[col]), lo, hi
    return float(r[col]), math.nan, math.nan


# Hình 1: tăng trưởng KG theo số bài
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
rnd = GROWTH[GROWTH.order != "papers.json"].groupby("k")
ref = GROWTH[GROWTH.order == "papers.json"].set_index("k")
ax = axes[0]
for col, lab, c, dy, va in (("entities", "Entity", "#2a78d6", -10, "top"),
                            ("relations", "Quan hệ", "#eb6834", 10, "bottom")):
    ax.fill_between(rnd[col].min().index, rnd[col].min(), rnd[col].max(), color=c, alpha=0.15, lw=0)
    ax.plot(ref.index, ref[col], color=c, lw=2, label=lab)
    for k in KS:   # số liệu của KG dựng thật (đường cong là phép chiếu từ KG30, lệch < 0,3%)
        ax.plot(k, ref.loc[k, col], "o", ms=8, color=c, mec=SURF, mew=2)
        ax.annotate(f"{KQ['kg'][str(k)][col]:,}".replace(",", "."), (k, ref.loc[k, col]),
                    textcoords="offset points", xytext=(6, dy), ha="left", va=va, fontsize=8.5, color=INK2)
ax.set_title("(a) Số entity và quan hệ theo số bài")
ax.set_xlabel("Số bài báo trong KG (k)")
ax.legend(frameon=False, loc="upper left")
grid(ax)
ax = axes[1]
for col, lab, c in (("entities_2plus_share", "Entity ở ≥ 2 bài", "#2a78d6"),
                    ("relations_2plus_share", "Quan hệ ở ≥ 2 bài", "#eb6834")):
    ax.fill_between(rnd[col].min().index, rnd[col].min() * 100, rnd[col].max() * 100, color=c, alpha=0.15, lw=0)
    ax.plot(ref.index, ref[col] * 100, color=c, lw=2, label=lab)
    for k in KS:
        ax.plot(k, ref.loc[k, col] * 100, "o", ms=8, color=c, mec=SURF, mew=2)
ax.set_title("(b) Liên kết giữa các bài")
ax.set_xlabel("Số bài báo trong KG (k)")
ax.set_ylabel("% entity / quan hệ")
ax.legend(frameon=False, loc="upper left")
grid(ax)
fig.text(0.5, -0.04, "Đường: thứ tự bài trong papers.json; vùng nhạt: min–max trên 20 thứ tự bài ngẫu nhiên",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout()
save(fig, "y3_fig1_growth.png")

# Hình 2: entity coverage và relation completeness
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
sets = [("in_scope", "Trong phạm vi KG"), ("S10", "S10 (10 bài đầu)"), ("all", "Toàn bộ 62 câu")]
for ax, (col, title) in zip(axes, (("entity_coverage", "(a) Entity coverage"),
                                   ("relation_completeness", "(b) Relation completeness"))):
    w = 0.26
    for j, k in enumerate(KS):
        xs = np.arange(len(sets)) + (j - 1) * w
        vals = [cov(s, k, col) for s, _ in sets]
        ax.bar(xs, [v[0] * 100 for v in vals], w - 0.03, color=CK[k], label=f"KG{k}")
        ax.errorbar(xs, [v[0] * 100 for v in vals], yerr=[[(v[0] - v[1]) * 100 for v in vals],
                                                          [(v[2] - v[0]) * 100 for v in vals]],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
        for x_, v in zip(xs, vals):
            ax.text(x_, 2, f"{v[0] * 100:.0f}", ha="center", va="bottom", fontsize=8,
                    color="white" if k > 10 else INK)
    ax.set_xticks(np.arange(len(sets)), [s[1] for s in sets])
    ax.set_title(title)
    ax.set_ylim(0, 105)
    grid(ax)
axes[0].set_ylabel("%")
axes[1].legend(frameon=False, ncol=3, loc="upper right")
fig.text(0.5, -0.03, "Thanh lỗi: khoảng tin cậy 95% (bootstrap trên các thực thể / quan hệ chuẩn)", ha="center",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
save(fig, "y3_fig2_coverage.png")

# Hình 3: năm chỉ số câu trả lời theo k (trong phạm vi KG, S10 cố định, toàn bộ 62 câu)
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
for ax, (m, title, as_pct) in zip(axes, METRICS):
    for j, (s, lab) in enumerate((("in_scope", "Trong phạm vi KG"), ("S10", "S10 cố định"), ("all", "Toàn bộ 62 câu"))):
        v = [stats["sets"][s][str(k)]["main"][m] for k in KS]
        f = 100 if as_pct else 1
        xs = np.array(KS) + (j - 1) * 0.7
        ax.errorbar(xs, [a[0] * f for a in v], yerr=[[(a[0] - a[1]) * f for a in v], [(a[2] - a[0]) * f for a in v]],
                    color=CS[s], lw=2, marker="o", ms=6, mec=SURF, mew=1.5, capsize=2.5, elinewidth=1, label=lab)
    ax.set_xticks(KS, [f"k={k}" for k in KS])
    ax.set_title(title, fontsize=9.5)
    if as_pct:
        ax.set_ylabel("%")
    grid(ax)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
fig.tight_layout()
save(fig, "y3_fig3_answer.png")

# Hình 4: kết cục của 62 câu có đáp án theo k
fig, ax = plt.subplots(figsize=(9, 3.2))
parts = [("in_correct", "Trong phạm vi: đúng", "#2a78d6"), ("in_wrong", "Trong phạm vi: sai", "#86b6ef"),
         ("in_refused", "Trong phạm vi: từ chối", "#cde2fb"), ("out_refused", "Ngoài phạm vi: từ chối", "#d9d8d1"),
         ("out_correct", "Ngoài phạm vi: vẫn đúng", "#1baf7a"), ("out_wrong", "Ngoài phạm vi: sai", "#eb6834")]
for yi, k in enumerate(KS):
    left = 0
    for key, lab, c in parts:
        v = outcome[str(k)][key]
        if v:
            ax.barh(yi, v - 0.15, left=left + 0.075, color=c, height=0.6, label=lab if yi == 0 or k == 10 else None)
            if v >= 3:
                ax.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=8.5,
                        color="white" if c in ("#2a78d6", "#1baf7a", "#eb6834") else INK)
        left += v
ax.set_yticks(range(len(KS)), [f"KG{k}" for k in KS])
ax.invert_yaxis()
ax.set_xlabel("Số câu (62 câu có đáp án)")
h_, l_ = ax.get_legend_handles_labels()
seen = dict(zip(l_, h_))
ax.legend([seen[lab] for _, lab, _ in parts if lab in seen], [lab for _, lab, _ in parts if lab in seen],
          frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28))
grid(ax, "x")
fig.tight_layout()
save(fig, "y3_fig4_outcome.png")

# Hình 5: độ đúng câu trả lời theo độ phủ thực thể / quan hệ của câu hỏi
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, (key, labs, title) in zip(axes, (
        ("coverage_vs_answer", (("full", "Đủ mọi thực thể"), ("partial", "Thiếu ≥ 1 thực thể")),
         "(a) Theo entity coverage của câu"),
        ("relation_vs_answer", (("rel_ge_50", "≥ 50% quan hệ"), ("rel_lt_50", "< 50% quan hệ")),
         "(b) Theo relation completeness của câu"))):
    for j, (lab_key, lab) in enumerate(labs):
        d = stats[key][lab_key]
        v = d["answer_correctness"]
        ax.bar(j, v[0] * 100, 0.55, color=("#2a78d6", "#86b6ef")[j])
        ax.errorbar(j, v[0] * 100, yerr=[[(v[0] - v[1]) * 100], [(v[2] - v[0]) * 100]], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=3)
        ax.text(j, 3, f"{v[0] * 100:.0f}%\nn={d['n']}", ha="center", va="bottom", fontsize=8.5,
                color="white" if j == 0 else INK)
    ax.set_xticks([0, 1], [lab for _, lab in labs])
    ax.set_title(title)
    ax.set_ylim(0, 105)
    grid(ax)
axes[0].set_ylabel("Answer correctness (%)")
fig.text(0.5, -0.04, "Gộp các câu trong phạm vi KG ở cả ba k (một câu được tính ở mỗi KG chứa nó)", ha="center",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
save(fig, "y3_fig5_cov_answer.png")
print("saved", sorted(p.name for p in OUT.glob("y3_*.png")), "+ report/stats_y3.json")
