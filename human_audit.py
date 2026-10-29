"""Giai đoạn 2 (X1, Y1.1): chấm tay khoảng 100 câu trả lời của nội dung 1, 3, 4 để đối chiếu với LLM-judge
(nội dung 2 đã có 36 câu trả lời chấm tay bằng human_judge.py, được gộp vào bảng tổng hợp).

    python human_audit.py --make     # phiếu testset/human_audit/phieu_cham.html (+ key.json)
    #   -> mở phiếu bằng trình duyệt, chấm, bấm "Xuất JSON" -> lưu vào testset/human_audit/human_labels.json
    python human_audit.py --score    # so với judge -> results/human_audit/ (human_audit.md, agreement.csv, ...)

Mẫu (seed 42), chỉ câu có đáp án mà hệ không refuse:
  Y1 (results/k4):        40 câu single-hop của RAG thuần: 20 câu judge gắn hallucination + 20 câu không (Y1.1);
                          10 câu multi-hop: 5 RAG thuần + 5 Self-RAG;
  Y3 (results/kg_scale):  GraphRAG, 13 câu với KG10 + 12 câu với KG20;
  Y4 (results/compression): 25 câu của bốn chế độ base / kg / text / both (7 / 6 / 6 / 6).
Mỗi câu trả lời: từng claim (do judge tách, lấy từ scores) có được context hỗ trợ không — 1 có; 0 không có trong
context; 2 mâu thuẫn context — và câu trả lời có đúng so với ground truth không. Không chấm passage. Context hiển thị
là đúng context judge dùng để verify claim: Y1 = top-k ban đầu của câu hỏi gốc (giống nhau cho hai hệ); Y3 = mô tả KG
+ 4 passage; Y4 = context (đã nén) đưa vào LLM. Thứ tự xáo trộn, phiếu không ghi nội dung / hệ / chế độ và không có
điểm judge (nằm trong key.json). So với điểm judge đã gộp 3 lần chấm (scores*.jsonl dùng trong báo cáo), ở mức câu
trả lời vì judge không lưu verdict từng claim.
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
from evaluate_graphrag import TESTSET_FILES as KG_TESTSET_FILES
from human_judge import TEMPLATE, boot_agreement, landis_koch

DIR = config.PROJECT_DIR / "testset" / "human_audit"
SHEET, KEY, LABELS = DIR / "phieu_cham.html", DIR / "key.json", DIR / "human_labels.json"
OUT = config.RESULTS_DIR / "human_audit"
Y2_PER_ANSWER = config.RESULTS_DIR / "human_judge" / "per_answer.csv"   # 36 câu trả lời nội dung 2 đã chấm

R = config.RESULTS_DIR
# (nhóm, hệ / chế độ, file câu trả lời, file điểm, trường context judge dùng, có mô tả KG)
SOURCES = {
    "y1_rag": ("Y1", "langchain_rag", R / "k4" / "answers_langchain_rag.jsonl", R / "k4" / "scores_langchain_rag.jsonl",
               "contexts", False),
    # judge chấm Self-RAG trên top-k ban đầu của câu hỏi gốc = context của RAG thuần (retrieved_contexts của Self-RAG
    # là lần truy xuất cuối, khác top-k ban đầu khi đã viết lại truy vấn)
    "y1_selfrag": ("Y1", "langgraph_selfrag", R / "k4" / "answers_langgraph_selfrag.jsonl",
                   R / "k4" / "scores_langgraph_selfrag.jsonl", "initial", False),
    **{f"y3_k{k}": ("Y3", f"graphrag_kg{k}", R / "kg_scale" / f"k{k}" / "answers_graphrag.jsonl",
                    R / "kg_scale" / f"k{k}" / "scores_graphrag.jsonl", "contexts", True) for k in (10, 20)},
    **{f"y4_{m}": ("Y4", m, R / "compression" / m / "answers.jsonl", R / "compression" / m / "scores.jsonl",
                   "contexts", True) for m in ("base", "kg", "text", "both")},
}

GUIDE = [
    ("Mục đích", "Đối chiếu LLM-judge (llama3.1:8b) với người chấm trên khoảng 100 câu trả lời của nội dung 1, 3, 4. "
                 "Chấm độc lập: không xem điểm của judge và không tìm xem câu trả lời thuộc nội dung / hệ nào."),
    ("Cách chấm", "Mỗi ô màu vàng chọn một nút; ô đã chấm chuyển màu xám. Có thể ghi lý do. Nhãn tự lưu trong trình "
                  "duyệt này, nhưng nên bấm \"Xuất JSON\" sau mỗi buổi chấm để có bản lưu; \"Nhập JSON\" để chấm tiếp "
                  "từ file đã xuất (ví dụ trên máy khác). Không có phần chấm passage."),
    ("Ý j được hỗ trợ?", "Có (1) nếu ý này suy ra trực tiếp được từ CONTEXT (phần \"Mô tả KG\" nếu có + các đoạn văn). "
                         "Nếu không: chọn \"không có trong context\" (0) khi context không nhắc tới, kể cả khi ý đó "
                         "đúng theo hiểu biết của bạn; chọn \"mâu thuẫn context\" (2) khi context nói khác."),
    ("Câu trả lời đúng?", "Có nếu câu trả lời nêu được các sự kiện chính của GROUND TRUTH và không mâu thuẫn với nó. "
                          "Không xét context ở bước này."),
    ("Tô sáng từ khớp", "Chỉ để đọc nhanh: trùng từ không có nghĩa là được hỗ trợ, thiếu từ có thể do dùng từ đồng "
                        "nghĩa. Luôn đọc rồi mới chấm. Tắt bằng nút trên thanh tiêu đề."),
    ("Khi xong", "Bấm \"Xuất JSON\", lưu file human_labels.json vào thư mục testset/human_audit/ của repo, rồi chạy "
                 "python human_audit.py --score."),
    ("Ước lượng thời gian", "Khoảng 3 phút mỗi câu trả lời, khoảng 5 giờ cho cả phiếu; có thể chia nhiều buổi."),
]


NAMES = {"hallucination": "Có claim không được hỗ trợ (hallucination)",
         "correct": "Câu trả lời đúng (answer correctness)"}


def testsets() -> dict[str, dict]:
    y1 = {t["id"]: t for t in load_testset(config.TESTSET_FILES)}
    kg = {t["id"]: t for t in load_testset([f for f in KG_TESTSET_FILES if f.exists()])}
    return {"Y1": y1, "Y3": kg, "Y4": kg}


def answered(src: str, tests: dict, qtype: str | None = None) -> list[str]:
    group, _, _, sc_path, _, _ = SOURCES[src]
    sc = load_jsonl(sc_path)
    return sorted(i for i, s in sc.items() if i in tests[group] and tests[group][i]["type"] != "unanswerable"
                  and not s["refusal"] and s.get("claims") and (qtype is None or tests[group][i]["type"] == qtype))


def sample(rng: random.Random, tests: dict) -> list[tuple[str, str, str]]:
    """[(nhóm mẫu, nguồn, id câu hỏi)]."""
    sc_rag = load_jsonl(SOURCES["y1_rag"][3])
    single = answered("y1_rag", tests, "single")
    out = [("y1_single_halluc", "y1_rag", i)
           for i in rng.sample([i for i in single if sc_rag[i]["hallucination"] == 1], 20)]
    out += [("y1_single_ok", "y1_rag", i) for i in rng.sample([i for i in single if sc_rag[i]["hallucination"] == 0], 20)]
    for src in ("y1_rag", "y1_selfrag"):
        out += [("y1_multihop", src, i) for i in rng.sample(answered(src, tests, "multihop"), 5)]
    for src, n in (("y3_k10", 13), ("y3_k20", 12)):
        out += [("y3", src, i) for i in rng.sample(answered(src, tests), n)]
    for src, n in (("y4_base", 7), ("y4_kg", 6), ("y4_text", 6), ("y4_both", 6)):
        out += [("y4", src, i) for i in rng.sample(answered(src, tests), n)]
    return out


def make(force=False):
    if LABELS.exists() and not force:
        raise SystemExit(f"{LABELS} đã có (đã chấm). Dùng --force để tạo lại phiếu (nhãn cũ có thể không còn khớp).")
    rng = random.Random(config.SEED)
    tests = testsets()
    cases = sample(rng, tests)
    rng.shuffle(cases)
    data, key_cases = [], {}
    for n, (stratum, src, qid) in enumerate(cases, 1):
        group, system, ans_path, sc_path, ctx_field, has_kg = SOURCES[src]
        a, s, t = load_jsonl(ans_path)[qid], load_jsonl(sc_path)[qid], tests[group][qid]
        ctx = load_jsonl(SOURCES["y1_rag"][2])[qid]["contexts"] if ctx_field == "initial" else a[ctx_field]
        code = f"A{n:03d}"
        key_cases[code] = {"id": qid, "group": group, "system": system, "stratum": stratum, "type": t["type"],
                           "source": src, "claims": s["claims"]}
        data.append({"code": code, "question": t["question"], "reference": t["ground_truth"], "answer": a["answer"],
                     "kg": (a.get("kg_context") or None) if has_kg else None,
                     "passages": [{"source": c["source"], "page": c.get("page"),
                                   "text": re.sub(r"(?<!\n)\n(?!\n)", " ", c["text"])} for c in ctx],
                     "claims": s["claims"]})
    sheet_id = hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()[:12]
    payload = json.dumps({"sheet_id": sheet_id, "title": "Phiếu chấm tay · nội dung 1, 3, 4", "useful": False,
                          "cause": True, "guide": GUIDE, "cases": data}, ensure_ascii=False)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload.replace("</", "<\\/"), 1)
    DIR.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(html, encoding="utf-8")
    KEY.write_text(json.dumps({"seed": config.SEED, "sheet_id": sheet_id, "cases": key_cases}, ensure_ascii=False,
                              indent=1), encoding="utf-8")
    n_lab = sum(len(c["claims"]) + 1 for c in key_cases.values())
    by = pd.Series([c["stratum"] for c in key_cases.values()]).value_counts().to_dict()
    print(f"Đã tạo {SHEET} ({len(cases)} câu trả lời, {n_lab} ô cần chấm; {by}) và {KEY}")


# ---------------- So sánh với judge ----------------
def read_labels(path=LABELS) -> pd.DataFrame:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("sheet_id") != key["sheet_id"]:
        raise SystemExit(f"{path} không khớp phiếu hiện tại (sheet_id {d.get('sheet_id')} ≠ {key['sheet_id']}).")
    rows = []
    for x in d["labels"]:
        ok = (0, 1, 2) if x["kind"] == "claim" else (0, 1)
        if x["label"] is not None and x["label"] not in ok:
            raise SystemExit(f"Nhãn {x['code']} {x['kind']} {x['index']} phải thuộc {ok}, đang là {x['label']!r}")
        rows.append({"code": x["code"], "kind": x["kind"], "index": x["index"], "human": x["label"],
                     "note": x.get("note", ""), **{f: key["cases"][x["code"]][f]
                                                   for f in ("id", "group", "system", "stratum", "type", "source")}})
    return pd.DataFrame(rows)


def score(path=LABELS):
    lab = read_labels(path)
    missing = lab[lab.human.isna()]
    if len(missing):
        raise SystemExit(f"Còn {len(missing)} ô chưa chấm, ví dụ ở câu {sorted(set(missing.code))[:5]}.")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    sc = {src: load_jsonl(SOURCES[src][3]) for src in {c["source"] for c in key["cases"].values()}}

    per = []
    for code, c in key["cases"].items():
        g, s = lab[lab.code == code], sc[c["source"]][c["id"]]
        cl = g[g.kind == "claim"].human.astype(int)
        per.append({"code": code, "id": c["id"], "group": c["group"], "system": c["system"], "stratum": c["stratum"],
                    "type": c["type"], "n_claims": len(cl),
                    "n_absent": int((cl == 0).sum()), "n_contradict": int((cl == 2).sum()),
                    "faithfulness_human": float((cl == 1).mean()), "faithfulness_judge": s["faithfulness"],
                    # điểm judge có thể là NaN (verdict thiếu sau khi thử lại): boot_agreement bỏ các câu đó
                    "hallucination_human": int((cl != 1).any()), "hallucination_judge": s["hallucination"],
                    "correct_human": int(g[g.kind == "correct"].human.iloc[0]),
                    "correct_judge": s["answer_correctness"]})
    per = pd.DataFrame(per)

    # nội dung 2: 36 câu trả lời đã chấm tay (human_judge.py), chỉ câu không refuse như mẫu ở đây
    y2 = pd.read_csv(Y2_PER_ANSWER) if Y2_PER_ANSWER.exists() else pd.DataFrame()
    if len(y2):
        y2 = y2[y2.refusal == 0].assign(group="Y2")
    allp = pd.concat([per, y2[[c for c in y2.columns if c in per.columns]]], ignore_index=True) if len(y2) else per

    def agree(df: pd.DataFrame) -> dict:
        out = {}
        for m in NAMES:
            d = df[["code", f"{m}_human", f"{m}_judge"]].rename(columns={f"{m}_human": "human"})
            out[m] = boot_agreement(d, f"{m}_judge")
        f = df.dropna(subset=["faithfulness_human", "faithfulness_judge"])
        out["faithfulness"] = {"n": len(f), "mean_human": round(float(f.faithfulness_human.mean()), 3),
                               "mean_judge": round(float(f.faithfulness_judge.mean()), 3),
                               "mae": round(float((f.faithfulness_human - f.faithfulness_judge).abs().mean()), 3),
                               "pearson": round(float(f.faithfulness_human.corr(f.faithfulness_judge)), 3)}
        for m in ("hallucination", "correct"):
            out[f"direction_{m}"] = {"judge1_human0": int(((df[f"{m}_judge"] == 1) & (df[f"{m}_human"] == 0)).sum()),
                                     "judge0_human1": int(((df[f"{m}_judge"] == 0) & (df[f"{m}_human"] == 1)).sum())}
        return out

    groups = {g: agree(d) for g, d in allp.groupby("group")}
    groups["Y1–Y4"] = agree(allp)
    groups["Y1, Y3, Y4"] = agree(per)   # chỉ mẫu chấm lần này (nội dung 2 là lần chấm trước)
    rate = lambda d: {"n": len(d), **{f"{m}_{w}": round(float(d[f"{m}_{w}"].mean()), 3)
                                      for m in (*NAMES, "faithfulness") for w in ("human", "judge")}}
    by_type = {r: {t: rate(d) for t, d in df.groupby("type")}
               for r, df in (("Y1–Y4", allp), ("Y2", allp[allp.group == "Y2"]), ("Y1, Y3, Y4", per))}

    # Y1.1: 40 câu single-hop của RAG thuần, 20 câu judge gắn hallucination + 20 câu không
    y11 = per[per.stratum.isin(["y1_single_halluc", "y1_single_ok"])]
    flagged, clean = y11[y11.hallucination_judge == 1], y11[y11.hallucination_judge == 0]
    claims1 = lab[(lab.kind == "claim") & lab.code.isin(y11.code)]
    y1_1 = {"n_flagged": len(flagged), "flagged_human_halluc": int(flagged.hallucination_human.sum()),
            "flagged_judge_wrong": int((flagged.hallucination_human == 0).sum()),
            "flagged_absent_only": int(((flagged.n_absent > 0) & (flagged.n_contradict == 0)).sum()),
            "flagged_contradict": int((flagged.n_contradict > 0).sum()),
            "n_clean": len(clean), "clean_human_halluc": int(clean.hallucination_human.sum()),
            "claims": int(len(claims1)), "claims_absent": int((claims1.human == 0).sum()),
            "claims_contradict": int((claims1.human == 2).sum()),
            "claims_per_answer": round(float(y11.n_claims.mean()), 2)}

    rows = []
    for g, a in groups.items():
        for m, name in NAMES.items():
            v = a[m]
            rows.append({"nhóm": g, "nhãn": name, **v, "mức (Landis & Koch)": landis_koch(v["kappa"])})
    ag = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    lab.to_csv(OUT / "labels.csv", index=False, encoding="utf-8-sig")
    per.to_csv(OUT / "per_answer.csv", index=False, encoding="utf-8-sig")
    ag.to_csv(OUT / "agreement.csv", index=False, encoding="utf-8-sig")
    stats = {"n_answers": len(per), "n_answers_with_y2": len(allp),
             "by_group": allp.groupby("group").code.count().to_dict(), "groups": groups, "by_type": by_type,
             "y1_1": y1_1}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    fa = pd.DataFrame([{"nhóm": g, **a["faithfulness"]} for g, a in groups.items()])
    lines = ["# Người chấm so với LLM-judge (nội dung 1–4)\n",
             f"{len(per)} câu trả lời chấm mới (nội dung 1, 3, 4) + {len(allp) - len(per)} câu trả lời của nội dung 2 "
             f"(human_judge.py); chỉ câu có đáp án mà hệ không refuse. Judge `{config.JUDGE_MODEL}`, điểm gộp 3 lần chấm.\n",
             "## 1. Agreement ở mức câu trả lời\n", ag.to_markdown(index=False), "",
             "*CI 95%: bootstrap theo câu trả lời. κ theo Landis & Koch. Hallucination của người: câu trả lời có ít "
             "nhất một claim không được hỗ trợ (không có trong context hoặc mâu thuẫn context).*\n",
             "## 2. Faithfulness\n", fa.to_markdown(index=False), "",
             "## 3. Y1.1: 40 câu single-hop của RAG thuần\n",
             f"Judge gắn hallucination {y1_1['n_flagged']} câu: người đồng ý {y1_1['flagged_human_halluc']}, judge sai "
             f"{y1_1['flagged_judge_wrong']} (mọi claim được hỗ trợ); trong các câu người đồng ý, "
             f"{y1_1['flagged_absent_only']} câu chỉ có claim không có trong context, {y1_1['flagged_contradict']} câu "
             f"có claim mâu thuẫn context. Judge không gắn {y1_1['n_clean']} câu: người thấy có claim không được hỗ "
             f"trợ ở {y1_1['clean_human_halluc']} câu. Trên {y1_1['claims']} claim ({y1_1['claims_per_answer']} claim "
             f"mỗi câu): {y1_1['claims_absent']} không có trong context, {y1_1['claims_contradict']} mâu thuẫn context.\n"]
    (OUT / "human_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "human_audit.md").read_text(encoding="utf-8"))


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
