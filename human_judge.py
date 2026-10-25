"""Ý 2: human judge tay một mẫu stratified để đối chiếu với LLM-judge (llama3.1:8b).

    python human_judge.py --make     # tạo phiếu chấm testset/human_judge/phieu_cham.html (+ key.json)
    #   -> mở phiếu bằng trình duyệt, chấm, bấm "Xuất JSON" -> lưu file vào testset/human_judge/human_labels.json
    python human_judge.py --score    # so với judge -> results/human_judge/ (human_judge.md, agreement.csv, ...)

Mẫu: 9 câu single-hop + 9 câu multi-hop (seed 42) trong 62 câu có đáp án của ý 2; mỗi câu chấm câu trả lời của CẢ
HAI hệ (Vector RAG, GraphRAG) -> 36 câu trả lời. Thứ tự xáo trộn, phiếu không ghi tên hệ thống và không có điểm của
judge (mã -> câu hỏi / hệ thống nằm trong key.json). Human judge gán đúng ba loại nhãn mà judge gán:
  - passage i có giúp suy ra ground truth không       -> retrieval precision (judge: prompt USEFUL)
  - ý (claim) j có được context hỗ trợ không              -> faithfulness, hallucination (judge: VERIFY_CLAIMS)
    (claim là các ý judge đã tách ra từ câu trả lời trong results/kg30; context = đúng phần hệ đưa vào LLM)
  - câu trả lời có đúng so với ground truth không         -> answer correctness (judge: CORRECTNESS)
So với điểm judge GỐC (results/kg30, dùng trong báo cáo): % agreement + Cohen's κ (CI 95% bootstrap theo câu
trả lời), và chênh lệch GraphRAG − Vector RAG tính trên nhãn human judge.
Phiếu HTML tự lưu nhãn trong trình duyệt (localStorage) nhưng chỉ file JSON xuất ra mới được dùng để tính.
"""
import argparse
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd

import config
from evaluate import load_jsonl, load_testset
from evaluate_graphrag import TESTSET_FILES, gold_sources
from kg_build import load_papers

SRC = config.RESULTS_DIR / "kg30"
DIR = config.PROJECT_DIR / "testset" / "human_judge"
SHEET, KEY, LABELS = DIR / "phieu_cham.html", DIR / "key.json", DIR / "human_labels.json"
TEMPLATE = DIR / "template.html"   # khuôn phiếu chấm (HTML + JS); --make chèn dữ liệu vào /*__DATA__*/
OUT = config.RESULTS_DIR / "human_judge"
SYSTEMS = ("vector_rag", "graphrag")
N_PER_TYPE = 9

GUIDE = [
    ("Mục đích", "Đối chiếu LLM-judge (llama3.1:8b) với human judge trên 36 câu trả lời của nội dung 2. Chấm độc lập: "
                 "không xem điểm của judge (results/kg30) và không tìm xem câu trả lời là của hệ nào trước khi chấm xong."),
    ("Cách chấm", "Mỗi ô màu vàng chọn Có (1) hoặc Không (0); ô đã chấm chuyển màu xám. Có thể ghi lý do. Nhãn tự lưu "
                  "trong trình duyệt này, nhưng nên bấm \"Xuất JSON\" sau mỗi buổi chấm để có bản lưu; \"Nhập JSON\" "
                  "để chấm tiếp từ file đã xuất (ví dụ trên máy khác)."),
    ("Đoạn i hữu ích?", "Có nếu passage chứa thông tin giúp suy ra (một phần) GROUND TRUTH. Đoạn chỉ cùng chủ đề nhưng "
                        "không chứa sự kiện nào của ground truth = Không. Chấm từng đoạn riêng, không phụ thuộc thứ tự."),
    ("Ý j được hỗ trợ?", "Có nếu ý này suy ra trực tiếp được từ CONTEXT của câu trả lời (phần \"Mô tả KG\" nếu có + 4 đoạn "
                         "văn). Không nếu context nói khác hoặc không nhắc tới, kể cả khi ý đó đúng theo hiểu biết của bạn."),
    ("Câu trả lời đúng?", "Có nếu câu trả lời nêu được các sự kiện chính của GROUND TRUTH và không mâu thuẫn với nó; câu "
                          "refuse (\"I cannot find the answer...\") = Không. Không xét context ở bước này."),
    ("Tô sáng từ khớp", "Từ khoá của ground truth được tô trong mô tả KG và passage; dưới mỗi ý có danh sách từ của ý "
                        "và nơi xuất hiện (KG, Đ1–Đ4), từ không thấy bị gạch ngang. Chỉ để đọc nhanh: trùng từ không "
                        "có nghĩa là hữu ích / được hỗ trợ (đoạn có thể nhắc từ nhưng nói ý khác), thiếu từ cũng có "
                        "thể do context dùng từ đồng nghĩa. Luôn đọc rồi mới chấm. Tắt bằng nút trên thanh tiêu đề."),
    ("Khi xong","Bấm \"Xuất JSON\", lưu file human_labels.json vào thư mục testset/human_judge/ của repo, rồi chạy "
                 "python human_judge.py --score."),
    ("Ước lượng thời gian", "Khoảng 4–5 phút mỗi câu trả lời, 2,5–3 giờ cho cả phiếu."),
]


def cohen_kappa(a, b) -> float:
    """Cohen's κ cho hai dãy nhãn (bất kỳ số lớp)."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in labels))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def eligible() -> dict[str, list[dict]]:
    papers = set(load_papers(config.KG_K))
    items = [t for t in load_testset([f for f in TESTSET_FILES if f.exists()])
             if t["type"] != "unanswerable" and set(gold_sources(t)) <= papers]
    return {qt: sorted((t for t in items if t["type"] == qt), key=lambda t: t["id"]) for qt in ("single", "multihop")}


def make(force=False):
    if LABELS.exists() and not force:
        raise SystemExit(f"{LABELS} đã có (đã chấm). Dùng --force để tạo lại phiếu (nhãn cũ có thể không còn khớp).")
    rng = random.Random(config.SEED)
    by_type = eligible()
    sample = [t for qt in ("single", "multihop") for t in rng.sample(by_type[qt], N_PER_TYPE)]
    answers = {s: load_jsonl(SRC / f"answers_{s}.jsonl") for s in SYSTEMS}
    scores = {s: load_jsonl(SRC / f"scores_{s}.jsonl") for s in SYSTEMS}
    cases = [(t, s) for t in sample for s in SYSTEMS]
    rng.shuffle(cases)

    data, key_cases = [], {}
    for n, (t, s) in enumerate(cases, 1):
        code = f"H{n:02d}"
        a = answers[s][t["id"]]
        claims = scores[s][t["id"]].get("claims") or []
        key_cases[code] = {"id": t["id"], "type": t["type"], "system": s, "n_passages": len(a["contexts"]),
                           "claims": claims}
        data.append({"code": code, "question": t["question"], "reference": t["ground_truth"], "answer": a["answer"],
                     "kg": a.get("kg_context") or None,
                     # văn bản trích từ PDF xuống dòng theo khổ trang: nối lại, giữ dòng trống (ngắt đoạn)
                     "passages": [{"source": c["source"], "page": c.get("page"),
                                   "text": re.sub(r"(?<!\n)\n(?!\n)", " ", c["text"])} for c in a["contexts"]],
                     "claims": claims})
    sheet_id = hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()[:12]
    payload = json.dumps({"sheet_id": sheet_id, "guide": GUIDE, "cases": data}, ensure_ascii=False)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload.replace("</", "<\\/"), 1)
    DIR.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(html, encoding="utf-8")
    KEY.write_text(json.dumps({"seed": config.SEED, "source": "results/kg30", "sheet_id": sheet_id,
                               "cases": key_cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_lab = sum(c["n_passages"] + len(c["claims"]) + 1 for c in key_cases.values())
    print(f"Đã tạo {SHEET} ({len(cases)} câu trả lời, {n_lab} ô cần chấm) và {KEY}")


# ---------------- So sánh với judge ----------------
def read_labels(path=LABELS) -> pd.DataFrame:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("sheet_id") != key["sheet_id"]:
        raise SystemExit(f"{path} không khớp phiếu hiện tại (sheet_id {d.get('sheet_id')} ≠ {key['sheet_id']}).")
    rows = []
    for x in d["labels"]:
        if x["label"] not in (None, 0, 1):
            raise SystemExit(f"Nhãn {x['code']} {x['kind']} {x['index']} phải là 0 / 1, đang là {x['label']!r}")
        rows.append({"code": x["code"], "kind": x["kind"], "index": x["index"], "human": x["label"],
                     "note": x.get("note", ""), **{f: key["cases"][x["code"]][f] for f in ("id", "type", "system")}})
    return pd.DataFrame(rows)


def boot_agreement(df: pd.DataFrame, judge_col: str, n=2000):
    """% agreement + κ, CI 95% bootstrap lấy mẫu lại theo câu trả lời (các nhãn của cùng câu trả lời đi cùng nhau)."""
    df = df.dropna(subset=["human", judge_col])
    groups = [g for _, g in df.groupby("code")]
    a, b = df["human"].to_numpy(int), df[judge_col].to_numpy(int)
    rng = np.random.default_rng(config.SEED)
    accs, kaps = [], []
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        x, y = s["human"].to_numpy(int), s[judge_col].to_numpy(int)
        accs.append(np.mean(x == y))
        kaps.append(cohen_kappa(x, y))
    return {"n_labels": len(df), "n_answers": len(groups), "human_pos": round(float(a.mean()), 3),
            "judge_pos": round(float(b.mean()), 3), "agreement": round(float(np.mean(a == b)), 3),
            "agreement_ci95": [round(float(v), 3) for v in np.percentile(accs, [2.5, 97.5])],
            "kappa": round(cohen_kappa(a, b), 3),
            "kappa_ci95": [round(float(v), 3) for v in np.nanpercentile(kaps, [2.5, 97.5])]}


def landis_koch(k: float) -> str:
    return next(name for lim, name in ((0, "kém"), (0.2, "rất thấp"), (0.4, "thấp"), (0.6, "trung bình"),
                                       (0.8, "khá"), (1.01, "gần như hoàn toàn")) if k < lim)


def score(path=LABELS):
    lab = read_labels(path)
    missing = lab[lab.human.isna()]
    if len(missing):
        raise SystemExit(f"Còn {len(missing)} ô chưa chấm, ví dụ ở câu {sorted(set(missing.code))[:5]}.")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    sc = {s: load_jsonl(SRC / f"scores_{s}.jsonl") for s in SYSTEMS}

    def judge_label(r):
        s = sc[r.system][r.id]
        if r.kind == "useful":
            return s["useful"][r["index"]]
        if r.kind == "correct":
            return s["answer_correctness"]
        return np.nan  # judge gốc không lưu verdict từng claim, chỉ lưu số claim không được hỗ trợ

    lab["judge"] = lab.apply(judge_label, axis=1)

    # mức câu trả lời: retrieval precision, faithfulness, hallucination, correctness của human judge và của judge
    per = []
    for code, c in key["cases"].items():
        g, s = lab[lab.code == code], sc[c["system"]][c["id"]]
        cl = g[g.kind == "claim"].human
        per.append({"code": code, "id": c["id"], "type": c["type"], "system": c["system"],
                    "refusal": s["refusal"],
                    "context_precision_human": g[g.kind == "useful"].human.mean(),
                    "context_precision_judge": s["context_precision"],
                    "faithfulness_human": cl.mean() if len(cl) and not s["refusal"] else np.nan,
                    "faithfulness_judge": s["faithfulness"],
                    "hallucination_human": int((cl == 0).any()) if not s["refusal"] else 0,
                    "hallucination_judge": s["hallucination"],
                    "correct_human": int(g[g.kind == "correct"].human.iloc[0]),
                    "correct_judge": s["answer_correctness"]})
    per = pd.DataFrame(per)

    agree = {"Passage hữu ích (retrieval precision)": boot_agreement(lab[lab.kind == "useful"], "judge"),
             "Câu trả lời đúng (answer correctness)": boot_agreement(lab[lab.kind == "correct"], "judge")}
    h = per[per.refusal == 0].rename(columns={"hallucination_human": "human", "hallucination_judge": "judge"})
    agree["Có ý không được hỗ trợ (hallucination)"] = boot_agreement(h, "judge")
    ag = pd.DataFrame([{"nhãn": k, **v} for k, v in agree.items()])
    ag["mức (Landis & Koch)"] = ag.kappa.map(landis_koch)

    f = per.dropna(subset=["faithfulness_human", "faithfulness_judge"])
    faith = {"n": len(f), "mae": round(float((f.faithfulness_human - f.faithfulness_judge).abs().mean()), 3),
             "pearson": round(float(f.faithfulness_human.corr(f.faithfulness_judge)), 3),
             "mean_human": round(float(f.faithfulness_human.mean()), 3),
             "mean_judge": round(float(f.faithfulness_judge.mean()), 3)}

    # Nguồn chênh lệch: chiều lệch (human 0 / judge 1 và ngược lại), tỉ lệ "có" theo loại câu, hệ, vị trí passage
    u, nr = lab[lab.kind == "useful"], per[per.refusal == 0]
    r3 = lambda v: round(float(v), 3)
    direction = {"useful": (u, "human", "judge"), "correct": (per, "correct_human", "correct_judge"),
                 "hallucination": (nr, "hallucination_human", "hallucination_judge")}
    breakdown = {
        "direction": {k: {"judge1_human0": int(((d[j] == 1) & (d[hu] == 0)).sum()),
                          "judge0_human1": int(((d[j] == 0) & (d[hu] == 1)).sum())}
                      for k, (d, hu, j) in direction.items()},
        "correct_by_type": {t: {"human": r3(g.correct_human.mean()), "judge": r3(g.correct_judge.mean()),
                                "judge1_human0": int(((g.correct_judge == 1) & (g.correct_human == 0)).sum())}
                            for t, g in per.groupby("type")},
        "useful_by_system": {s: {"human": r3(g.human.mean()), "judge": r3(g.judge.mean())}
                             for s, g in u.groupby("system")},
        "useful_by_index": {int(i) + 1: {"human": r3(g.human.mean()), "judge": r3(g.judge.mean())}
                            for i, g in u.groupby("index")},
        "faithfulness_by_system": {s: {"human": r3(g.faithfulness_human.mean()), "judge": r3(g.faithfulness_judge.mean()),
                                       "judge1_human0": int(((g.hallucination_judge == 1)
                                                             & (g.hallucination_human == 0)).sum())}
                                   for s, g in f.groupby("system")},
    }

    # GraphRAG − Vector RAG trên cùng câu hỏi: theo nhãn human judge và theo judge (cùng mẫu)
    diffs = []
    for m in ("context_precision", "faithfulness", "hallucination", "correct"):
        for who in ("human", "judge"):
            p = per.pivot(index="id", columns="system", values=f"{m}_{who}").dropna()
            d = (p["graphrag"] - p["vector_rag"]).to_numpy(float)
            rng = np.random.default_rng(config.SEED)
            bs = rng.choice(d, size=(5000, len(d))).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            diffs.append({"metric": m, "who": who, "n_pairs": len(d), "graphrag": round(float(p["graphrag"].mean()), 3),
                          "vector_rag": round(float(p["vector_rag"].mean()), 3), "diff": round(float(d.mean()), 3),
                          "ci95": [round(float(lo), 3), round(float(hi), 3)], "significant": bool(lo > 0 or hi < 0)})
    diffs = pd.DataFrame(diffs)

    OUT.mkdir(parents=True, exist_ok=True)
    lab.to_csv(OUT / "labels.csv", index=False, encoding="utf-8-sig")
    per.to_csv(OUT / "per_answer.csv", index=False, encoding="utf-8-sig")
    ag.to_csv(OUT / "agreement.csv", index=False, encoding="utf-8-sig")
    diffs.to_csv(OUT / "paired_diff.csv", index=False, encoding="utf-8-sig")
    stats = {"n_answers": len(per), "n_questions": per.id.nunique(), "by_type": per.groupby("type").id.nunique().to_dict(),
             "agreement": {k: v for k, v in agree.items()}, "faithfulness": faith, "breakdown": breakdown,
             "paired": diffs.to_dict(orient="records")}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Human judge so với LLM-judge (ý 2)\n",
             f"{len(per)} câu trả lời ({per.id.nunique()} câu hỏi × 2 hệ; single-hop {stats['by_type'].get('single')}, "
             f"multi-hop {stats['by_type'].get('multihop')} câu hỏi), judge gốc `{config.JUDGE_MODEL}` (results/kg30).\n",
             "## 1. Mức agreement\n", ag.to_markdown(index=False), "",
             "*CI 95%: bootstrap theo câu trả lời. κ theo Landis & Koch: < 0,2 rất thấp, 0,2–0,4 thấp, 0,4–0,6 trung bình, "
             "0,6–0,8 khá, > 0,8 gần như hoàn toàn. Hallucination chỉ tính câu không refuse; judge gốc không lưu verdict "
             "từng ý nên so ở mức câu trả lời. Khi một nhãn chiếm phần lớn (human_pos / judge_pos gần 0 hoặc 1, vd. answer "
             "correctness), κ có thể thấp dù % agreement cao (prevalence paradox), nên đọc κ cùng % agreement.*\n",
             f"Faithfulness (mức câu trả lời, n = {faith['n']}): trung bình human judge {faith['mean_human']}, judge "
             f"{faith['mean_judge']}; sai khác tuyệt đối trung bình {faith['mae']}; tương quan Pearson {faith['pearson']}.\n",
             "## 2. GraphRAG − Vector RAG trên cùng mẫu: nhãn human judge và nhãn judge\n", diffs.to_markdown(index=False), "",
             "## 3. Nguồn chênh lệch\n",
             "Chiều lệch (số nhãn / câu trả lời): " + "; ".join(
                 f"{k}: judge 1 – human 0 = {v['judge1_human0']}, judge 0 – human 1 = {v['judge0_human1']}"
                 for k, v in breakdown["direction"].items()) + "\n",
             "Answer correctness theo loại câu: " + "; ".join(
                 f"{t}: human {v['human']}, judge {v['judge']} ({v['judge1_human0']} câu judge 1 – human 0)"
                 for t, v in breakdown["correct_by_type"].items()) + "\n",
             "Passage hữu ích theo hệ: " + "; ".join(f"{s}: human {v['human']}, judge {v['judge']}"
                                                     for s, v in breakdown["useful_by_system"].items())
             + "; theo vị trí: " + "; ".join(f"#{i}: human {v['human']}, judge {v['judge']}"
                                             for i, v in breakdown["useful_by_index"].items()) + "\n",
             "Faithfulness theo hệ: " + "; ".join(
                 f"{s}: human {v['human']}, judge {v['judge']} ({v['judge1_human0']} câu judge có ý không được hỗ trợ "
                 f"– human không)" for s, v in breakdown["faithfulness_by_system"].items()) + "\n"]
    (OUT / "human_judge.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "human_judge.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true", help="tạo lại phiếu dù đã có human_labels.json")
    ap.add_argument("--labels", default=str(LABELS), help="file JSON xuất từ phiếu chấm")
    args = ap.parse_args()
    if args.make:
        make(args.force)
    elif args.score:
        from pathlib import Path
        score(Path(args.labels))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
