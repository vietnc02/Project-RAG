"""Giai đoạn 2, việc 3 (Y4.4, Y4.5): chấm tay cặp câu trả lời context gốc / nén cả hai của nội dung 4.

    python human_audit_y4.py --make     # phiếu testset/human_audit_y4/phieu_cham.html (+ key.json)
    #   -> chấm, "Xuất JSON" -> lưu thành testset/human_audit_y4/human_labels.json
    python human_audit_y4.py --make-where   # phiếu chấm lại riêng các ô "ý nằm ở đâu" (gt:*:A, gt:*:B)
    #   -> testset/human_audit_y4/phieu_y_lai.html; "Xuất JSON" -> testset/human_audit_y4/human_labels_where.json
    python human_audit_y4.py --score    # -> results/human_audit_y4/ (human_audit_y4.md, stats.json, per_answer.csv)
    #   nếu có human_labels_where.json thì nhãn gt:*:A / gt:*:B lấy từ file này (thay cho human_labels.json)

Mẫu (seed 42): 30 câu có đáp án mà cả hai chế độ (base, both) đều trả lời. Mỗi câu hiện hai câu trả lời A / B (thứ tự
ngẫu nhiên, không ghi chế độ), mỗi câu trả lời kèm đúng context đưa vào LLM (mô tả KG + passage, đã nén nếu là both).
  - Y4.4: từng claim của A, B (claim do judge tách, scores.jsonl): 1 có trong context, 0 không có, 2 mâu thuẫn context
    -> faithfulness theo người; so chênh lệch both − base theo người với theo judge, và độ lệch judge − người ở context
    dài (base) so với context đã nén (both) để kiểm tra thiên lệch của judge theo độ dài context.
  - Y4.5: từng ý của ground truth (gt_claims.jsonl, judge tách, dùng cho context recall): ý bắt buộc không (1 / 0); với
    A và B: 3 có trong context và câu trả lời, 2 chỉ trong context, 1 chỉ trong câu trả lời, 0 không có
    -> context recall và answer recall theo người (mọi ý / chỉ ý bắt buộc); "đúng" = mọi ý bắt buộc có trong câu trả lời.
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
from evaluate_graphrag import TESTSET_FILES

DIR = config.PROJECT_DIR / "testset" / "human_audit_y4"
SHEET, KEY, LABELS = DIR / "phieu_cham.html", DIR / "key.json", DIR / "human_labels.json"
SHEET_W, KEY_W, LABELS_W = DIR / "phieu_y_lai.html", DIR / "key_where.json", DIR / "human_labels_where.json"
TEMPLATE = config.PROJECT_DIR / "testset" / "review" / "template.html"
RUN = config.RESULTS_DIR / "compression"
OUT = config.RESULTS_DIR / "human_audit_y4"
MODES = ("base", "both")
N = 30

CLAIM = [[1, "Có (1)", "y"], [0, "Không: không có trong context (0)", "n"], [2, "Không: mâu thuẫn context (2)", "n"]]
MUST = [[1, "Bắt buộc (1)", "y"], [0, "Không bắt buộc (0)", "n"]]
WHERE = [[3, "Context và câu trả lời (3)", "y"], [2, "Chỉ context (2)", ""], [1, "Chỉ câu trả lời (1)", ""],
         [0, "Không có (0)", "n"]]
GUIDE = [
    ("Mục đích", "Kiểm tra judge ở nội dung 4: faithfulness của câu trả lời khi context đầy đủ và khi context bị nén, và "
                 "các ý quan trọng của đáp án còn trong context / câu trả lời không. Hai câu trả lời A, B của cùng một câu "
                 "hỏi, thứ tự ngẫu nhiên, không ghi chế độ; không tìm xem A, B là chế độ nào."),
    ("Claim của A / B", "Có (1) nếu claim suy ra trực tiếp được từ CONTEXT CỦA CHÍNH câu trả lời đó (khung context ngay "
                        "trên). Không có trong context (0) nếu context không nhắc tới, kể cả khi claim đúng theo hiểu "
                        "biết của bạn; mâu thuẫn context (2) nếu context nói khác."),
    ("Ý của ground truth", "Mỗi ý: (1) có phải ý bắt buộc để trả lời đúng câu hỏi không (thường 1–3 ý; ý phụ, chi tiết "
                           "thêm = không bắt buộc); (2) với A và với B: ý đó có trong context, trong câu trả lời, cả hai "
                           "hay không có. Chấm cả ý không bắt buộc."),
    ("Khi xong", "Bấm \"Xuất JSON\", lưu file vào testset/human_audit_y4/human_labels.json, rồi chạy "
                 "python human_audit_y4.py --score."),
    ("Ước lượng thời gian", "Khoảng 8–10 phút mỗi câu hỏi, 4–5 giờ cho cả phiếu; có thể chia nhiều buổi."),
]


def context_text(a: dict) -> tuple[str, str]:
    kg = a.get("kg_context") or ""
    ps = "\n\n".join(f"[Đoạn {i}] {c['source']} · trang {c.get('page')}\n" + re.sub(r"(?<!\n)\n(?!\n)", " ", c["text"])
                     for i, c in enumerate(a["contexts"], 1))
    return kg, ps


def make(force=False):
    if LABELS.exists() and not force:
        raise SystemExit(f"{LABELS} đã có (đã chấm). Dùng --force để tạo lại phiếu.")
    items = {t["id"]: t for t in load_testset([f for f in TESTSET_FILES if f.exists()])}
    ans = {m: load_jsonl(RUN / m / "answers.jsonl") for m in MODES}
    sc = {m: load_jsonl(RUN / m / "scores.jsonl") for m in MODES}
    gt = load_jsonl(RUN / "gt_claims.jsonl")
    ids = sorted(i for i, t in items.items() if t["type"] != "unanswerable" and gt.get(i, {}).get("claims")
                 and all(not sc[m][i]["refusal"] and sc[m][i].get("claims") for m in MODES))
    rng = random.Random(config.SEED)
    sample = rng.sample(ids, N)
    cases, key = [], {}
    for n, qid in enumerate(sample, 1):
        order = list(MODES)
        rng.shuffle(order)
        code = f"C{n:02d}"
        key[code] = {"id": qid, "type": items[qid]["type"], "A": order[0], "B": order[1],
                     "claims": {s: sc[m][qid]["claims"] for s, m in zip("AB", order)}, "gt": gt[qid]["claims"]}
        blocks = [{"title": "Câu hỏi", "text": items[qid]["question"]},
                  {"title": "Ground truth", "text": items[qid]["ground_truth"]}]
        for s, m in zip("AB", order):
            kg, ps = context_text(ans[m][qid])
            blocks += [{"title": f"Câu trả lời {s}", "text": ans[m][qid]["answer"]},
                       {"title": f"Context {s}: mô tả KG", "text": kg, "kind": "kg"},
                       {"title": f"Context {s}: các đoạn văn", "text": ps}]
            blocks += [{"key": f"{s}:claim:{j}", "group": f"Claim của câu trả lời {s}: có được context {s} hỗ trợ?",
                        "text": f"Ý {j + 1}. {cl}", "prompt": "Được context hỗ trợ?", "options": CLAIM}
                       for j, cl in enumerate(sc[m][qid]["claims"])]
        for i, g in enumerate(gt[qid]["claims"]):
            grp = "Các ý của ground truth"
            blocks += [{"key": f"gt:{i}:must", "group": grp, "text": f"Ý {i + 1}. {g}", "prompt": "Ý bắt buộc?",
                        "options": MUST}]
            blocks += [{"key": f"gt:{i}:{s}", "group": grp, "prompt": f"Với câu trả lời {s}: ý này có ở đâu?",
                        "options": WHERE} for s in "AB"]
        cases.append({"code": code, "title": f"câu hỏi {n}/{N}", "blocks": blocks})
    sheet_id = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()[:12]
    payload = json.dumps({"sheet_id": sheet_id, "title": "Phiếu chấm tay · nội dung 4 (context gốc / nén)",
                          "file": "human_labels.json", "guide": GUIDE, "cases": cases}, ensure_ascii=False)
    DIR.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload.replace("</", "<\\/"), 1),
                     encoding="utf-8")
    KEY.write_text(json.dumps({"seed": config.SEED, "sheet_id": sheet_id, "cases": key}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    n_lab = sum(len(c["claims"]["A"]) + len(c["claims"]["B"]) + 3 * len(c["gt"]) for c in key.values())
    print(f"Đã tạo {SHEET} ({N} câu hỏi, {n_lab} ô cần chấm) và {KEY}")


GUIDE_W = [
    ("Mục đích", "Chấm lại các ô \"ý nằm ở đâu\" của phiếu nội dung 4. Lần trước các ô này chỉ được chấm \"Chỉ context\" "
                 "hoặc \"Không có\", tức là đã chỉ xét context mà bỏ qua câu trả lời. Các nhãn claim và \"ý bắt buộc\" giữ "
                 "nguyên, không chấm lại."),
    ("Cách chấm mỗi ô", "Với từng ý của ground truth và từng câu trả lời (A, B), xét hai câu hỏi riêng: (a) CÂU TRẢ LỜI "
                        "có nêu ý này không (cùng nghĩa, có thể khác chữ)? (b) CONTEXT của chính câu trả lời đó có chứa "
                        "ý này không? Cả hai có → 3; chỉ context → 2; chỉ câu trả lời → 1; không có → 0."),
    ("Ví dụ", "Ý \"Ring signatures can be used in a blockchain-IoT environment\"; câu trả lời: \"Monero can be used to "
              "obscure the information of the sender ... using ring signatures in a blockchain-IoT environment\"; context "
              "có nhắc ring signature → 3 (không phải 2)."),
    ("Mẹo", "Đọc câu trả lời trước (ngắn), đánh dấu các ý nó đã nêu; chỉ mở context khi cần quyết định giữa 3/1 hoặc 2/0."),
    ("Khi xong", "Bấm \"Xuất JSON\", lưu file vào testset/human_audit_y4/human_labels_where.json, rồi chạy "
                 "python human_audit_y4.py --score."),
    ("Ước lượng thời gian", "Khoảng 3–4 phút mỗi câu hỏi, 1,5–2 giờ cho cả phiếu."),
]


def make_where():
    """Phiếu chấm lại các ô gt:*:A / gt:*:B: cùng mã mục, cùng A/B và cùng context với phiếu chính."""
    key = json.loads(KEY.read_text(encoding="utf-8"))
    items = {t["id"]: t for t in load_testset([f for f in TESTSET_FILES if f.exists()])}
    ans = {m: load_jsonl(RUN / m / "answers.jsonl") for m in MODES}
    cases = []
    for code, c in key["cases"].items():
        qid = c["id"]
        blocks = [{"title": "Câu hỏi", "text": items[qid]["question"]},
                  {"title": "Ground truth", "text": items[qid]["ground_truth"]}]
        for s in "AB":
            kg, ps = context_text(ans[c[s]][qid])
            blocks += [{"title": f"Câu trả lời {s}", "text": ans[c[s]][qid]["answer"]},
                       {"title": f"Context {s}: mô tả KG", "text": kg, "kind": "kg"},
                       {"title": f"Context {s}: các đoạn văn", "text": ps}]
        for i, g in enumerate(c["gt"]):
            blocks += [{"key": f"gt:{i}:{s}", "group": f"Ý {i + 1} của ground truth",
                        **({"text": f"Ý {i + 1}. {g}"} if s == "A" else {}),
                        "prompt": f"Câu trả lời {s}: ý này có trong câu trả lời {s}? có trong context {s}?",
                        "options": WHERE} for s in "AB"]
        cases.append({"code": code, "title": f"câu hỏi {code[1:]}/{len(key['cases'])}", "blocks": blocks})
    sheet_id = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()[:12]
    payload = json.dumps({"sheet_id": sheet_id, "title": "Phiếu chấm lại · nội dung 4 (ý nằm ở đâu)",
                          "file": "human_labels_where.json", "guide": GUIDE_W, "cases": cases}, ensure_ascii=False)
    SHEET_W.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload.replace("</", "<\\/"), 1),
                       encoding="utf-8")
    KEY_W.write_text(json.dumps({"sheet_id": sheet_id, "main_sheet_id": key["sheet_id"]}, indent=1), encoding="utf-8")
    print(f"Đã tạo {SHEET_W} ({len(cases)} câu hỏi, {sum(2 * len(c['gt']) for c in key['cases'].values())} ô cần chấm)")


def paired_ci(d, n=5000):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    bs = np.random.default_rng(config.SEED).choice(d, size=(n, len(d))).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(d), "delta": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def score(path=LABELS):
    key = json.loads(KEY.read_text(encoding="utf-8"))
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("sheet_id") != key["sheet_id"]:
        raise SystemExit(f"{path} không khớp phiếu hiện tại.")
    lab = {(x["code"], x["key"]): x["label"] for x in d["labels"]}
    if LABELS_W.exists():  # phiếu chấm lại các ô "ý nằm ở đâu" thay cho nhãn gt:*:A / gt:*:B của phiếu chính
        w = json.loads(LABELS_W.read_text(encoding="utf-8"))
        if w.get("sheet_id") != json.loads(KEY_W.read_text(encoding="utf-8"))["sheet_id"]:
            raise SystemExit(f"{LABELS_W} không khớp phiếu chấm lại hiện tại.")
        lab.update({(x["code"], x["key"]): x["label"] for x in w["labels"]})
        print(f"Nhãn \"ý nằm ở đâu\" lấy từ {LABELS_W.name}")
    if any(v is None for v in lab.values()):
        raise SystemExit(f"Còn {sum(v is None for v in lab.values())} ô chưa chấm.")
    sc = {m: load_jsonl(RUN / m / "scores.jsonl") for m in MODES}
    rows = []
    for code, c in key["cases"].items():
        must = [lab[(code, f"gt:{i}:must")] for i in range(len(c["gt"]))]
        for s in "AB":
            m, qid = c[s], c["id"]
            cl = [lab[(code, f"{s}:claim:{j}")] for j in range(len(c["claims"][s]))]
            where = [lab[(code, f"gt:{i}:{s}")] for i in range(len(c["gt"]))]
            in_ctx = [w in (2, 3) for w in where]
            in_ans = [w in (1, 3) for w in where]
            mi = [i for i, x in enumerate(must) if x == 1]
            rows.append({"code": code, "id": qid, "type": c["type"], "mode": m,
                         "faithfulness_human": float(np.mean([x == 1 for x in cl])),
                         "faithfulness_judge": sc[m][qid]["faithfulness"],
                         "hallucination_human": int(any(x != 1 for x in cl)),
                         "hallucination_judge": sc[m][qid]["hallucination"],
                         "n_contradict": int(sum(x == 2 for x in cl)),
                         "context_recall_human": float(np.mean(in_ctx)),
                         "context_recall_judge": sc[m][qid]["context_recall"],
                         "n_must": len(mi),
                         "context_recall_must_human": float(np.mean([in_ctx[i] for i in mi])) if mi else np.nan,
                         "answer_recall_must_human": float(np.mean([in_ans[i] for i in mi])) if mi else np.nan,
                         "correct_must_human": int(all(in_ans[i] for i in mi)) if mi else np.nan,
                         "correct_judge": sc[m][qid]["answer_correctness"],
                         "answer_only_ideas": int(sum(w == 1 for w in where))})
    per = pd.DataFrame(rows)
    pv = lambda col: per.pivot(index="id", columns="mode", values=col)
    metrics = ["faithfulness_human", "faithfulness_judge", "context_recall_human", "context_recall_judge",
               "context_recall_must_human", "answer_recall_must_human", "correct_must_human", "correct_judge",
               "hallucination_human", "hallucination_judge"]
    means = {mt: {m: round(float(per[per["mode"] == m][mt].mean()), 4) for m in MODES} for mt in metrics}
    paired = {mt: paired_ci(pv(mt)["both"] - pv(mt)["base"]) for mt in metrics}
    # thiên lệch theo độ dài context: độ lệch judge − người ở context gốc (dài) so với context nén (ngắn)
    gap = pv("faithfulness_judge") - pv("faithfulness_human")
    length_bias = {"gap_base": round(float(gap["base"].mean()), 4), "gap_both": round(float(gap["both"].mean()), 4),
                   "gap_both_minus_base": paired_ci(gap["both"] - gap["base"])}
    musts = [lab[(code, f"gt:{i}:must")] for code, c in key["cases"].items() for i in range(len(c["gt"]))]
    stats = {"n_questions": len(key["cases"]), "n_gt_ideas": len(musts), "n_must": int(sum(musts)),
             "must_per_question": round(sum(musts) / len(key["cases"]), 2), "means": means, "paired": paired,
             "length_bias": length_bias,
             "correct_agreement": round(float((per.correct_must_human == per.correct_judge).mean()), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "per_answer.csv", index=False, encoding="utf-8-sig")
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    f = lambda x: f"{x['delta']:+.3f} [{x['ci95'][0]:+.3f}; {x['ci95'][1]:+.3f}] (n = {x['n']})"
    tab = pd.DataFrame([{"chỉ số": mt, "base": means[mt]["base"], "both": means[mt]["both"], "both − base": f(paired[mt])}
                        for mt in metrics])
    lines = ["# Chấm tay nội dung 4: context gốc (base) và nén cả hai (both)\n",
             f"{stats['n_questions']} câu hỏi × 2 chế độ; {stats['n_gt_ideas']} ý ground truth, {stats['n_must']} ý bắt "
             f"buộc ({stats['must_per_question']} mỗi câu). Judge `{config.JUDGE_MODEL}`, điểm gộp 3 lần chấm.\n",
             tab.to_markdown(index=False), "",
             f"Độ lệch faithfulness judge − người: context gốc {length_bias['gap_base']:+.3f}, context nén "
             f"{length_bias['gap_both']:+.3f}; chênh lệch nén − gốc {f(length_bias['gap_both_minus_base'])}.\n",
             f"Khớp giữa \"đúng theo ý bắt buộc\" (người) và answer correctness (judge): "
             f"{stats['correct_agreement']:.1%}.\n"]
    (OUT / "human_audit_y4.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "human_audit_y4.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--make-where", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--labels", default=str(LABELS))
    args = ap.parse_args()
    if args.make:
        make(args.force)
    elif args.make_where:
        make_where()
    elif args.score:
        from pathlib import Path
        score(Path(args.labels))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
