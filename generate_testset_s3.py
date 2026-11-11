"""Giai đoạn 3 (Y3.1, Y4.1): sinh thêm câu hỏi có đáp án trong 30 bài dựng KG (rồi DUYỆT TAY).

    python generate_testset_s3.py --single      # single-hop -> testset/singlehop_s3_raw.json (chạy tiếp được)
    python generate_testset_s3.py --multihop 60 # multi-hop trong 30 bài -> testset/multihop_s3_raw.json
    python generate_testset_s3.py --sheet       # phiếu duyệt testset/review_s3/phieu_duyet.html (+ key.json)
    python generate_testset_s3.py --apply       # nhãn duyệt (review_s3/human_labels.json) -> singlehop_s3.json,
                                                #   multihop_s3.json (chỉ câu giữ lại, kèm câu đã sửa)

Mục tiêu: S10 (câu về 10 bài đầu) từ 15 lên khoảng 75 câu (Y3.1) và nội dung 4 có ít nhất 250 câu có đáp án (Y4.1).
Single-hop dùng đúng prompt và bộ lọc của generate_testset.py; mỗi bài chia các đoạn thành N khúc liên tiếp và lấy một
đoạn mỗi khúc (rải đều trong bài, N = 8 với 10 bài đầu, 7 với các bài còn lại), bỏ đoạn đã dùng cho câu hỏi cũ và câu
gần trùng (cosine embedding câu hỏi >= 0,9) với câu đã có. Multi-hop dùng make_multihop của generate_testset.py, chỉ
trong 30 bài, không lặp cặp bài / đoạn của multihop.json, multihop_kg.json.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from ollama import ResponseError
from tqdm import tqdm

import config
from common import get_embeddings, get_llm
from generate_testset import PROMPT, good_chunk, make_multihop
from kg_build import load_chunks, load_papers, use_kg_store

DIR = config.TESTSET_FILE.parent
OLD_FILES = [config.TESTSET_FILE, *(DIR / f for f in ("singlehop_kg.json", "multihop.json", "multihop_kg.json"))]
SINGLE_RAW, MULTI_RAW = DIR / "singlehop_s3_raw.json", DIR / "multihop_s3_raw.json"
SINGLE, MULTI = DIR / "singlehop_s3.json", DIR / "multihop_s3.json"
RDIR = DIR / "review_s3"
SHEET, KEY, LABELS = RDIR / "phieu_duyet.html", RDIR / "key.json", RDIR / "human_labels.json"
TEMPLATE = DIR / "review" / "template.html"
PER_PAPER = {10: 8, 30: 7}   # số câu cần sinh mỗi bài: bài 1–10, bài 11–30
DUP_COS = 0.9


def per_paper(rank: int) -> int:
    return next(n for k, n in sorted(PER_PAPER.items()) if rank <= k)


def old_items() -> list[dict]:
    return [t for f in OLD_FILES if f.exists() for t in json.loads(f.read_text(encoding="utf-8"))]


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_single(seed: int, limit: int = 0):
    papers = load_papers()[: limit or None]
    old = old_items()
    used_chunks = {c for t in old for c in ([t["chunk_id"]] if "chunk_id" in t else t.get("chunk_ids", []))}
    by_paper = {p: [] for p in papers}
    for c in load_chunks():
        if c["source"] in by_paper and good_chunk(c) and c["chunk_id"] not in used_chunks and not re.search(
                r"^\s*(references|bibliography)\b", c["text"][:200], re.I | re.M):
            by_paper[c["source"]].append(c)
    items = json.loads(SINGLE_RAW.read_text(encoding="utf-8")) if SINGLE_RAW.exists() else []
    emb = get_embeddings()
    # câu đã có (cũ + đã sinh) của từng bài, để bỏ câu gần trùng
    qs = {p: [] for p in papers}
    for t in old + items:
        for s in ([t["source"]] if "source" in t else t.get("sources", [])):
            if s in qs:
                qs[s].append(t["question"])
    qv = {p: list(unit(emb.embed_documents(v))) if v else [] for p, v in qs.items()}
    chain = PROMPT | get_llm(json_mode=True) | StrOutputParser()
    rng = random.Random(seed)
    for rank, p in enumerate(tqdm(papers, desc="Sinh câu single-hop"), 1):
        n, have = per_paper(rank), sum(t["source"] == p for t in items)
        chunks = sorted(by_paper[p], key=lambda c: int(c["chunk_id"].rsplit("-", 1)[1]))   # theo thứ tự trong bài
        segs = [s for s in np.array_split(np.arange(len(chunks)), n) if len(s)]
        done_segs = {t["segment"] for t in items if t["source"] == p}
        for si, seg in enumerate(segs):
            if have >= n or si in done_segs:
                continue
            for i in rng.sample(list(seg), min(3, len(seg))):   # thử tối đa 3 đoạn mỗi khúc
                c = chunks[i]
                try:
                    d = json.loads(chain.invoke({"title": p, "text": c["text"]}))
                except (json.JSONDecodeError, ResponseError):  # JSON hỏng / Ollama huỷ do lặp token
                    continue
                if d.get("skip") or not d.get("question") or not d.get("ground_truth"):
                    continue
                # câu hỏi phải tự đứng được (như bộ lọc multi-hop của generate_testset.py)
                if re.search(r"\bpassage\b|\bthis (paper|study|work|article|text)\b|\bthe authors\b|"
                             r"\baccording to the (paper|article|text|study)\b", d["question"], re.I):
                    continue
                v = unit(emb.embed_query(d["question"]))
                if qv[p] and max(float(v @ u) for u in qv[p]) >= DUP_COS:
                    continue
                qv[p].append(v)
                items.append({"id": f"t{len(items) + 1:03d}", "type": "single", "question": d["question"].strip(),
                              "ground_truth": d["ground_truth"].strip(), "source": p, "paper_rank": rank,
                              "segment": si, "topic": c["topic"], "page": c["page"], "chunk_id": c["chunk_id"],
                              "chunk_text": c["text"], "reviewed": False, "keep": True})
                have += 1
                SINGLE_RAW.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")  # lưu dần
                break
    print(f"{len(items)} câu single-hop -> {SINGLE_RAW}")


def make_multi(n: int, seed: int):
    use_kg_store()
    random.seed(seed)
    papers = load_papers()
    by_paper = {p: [] for p in papers}
    for c in load_chunks():
        if c["source"] in by_paper and good_chunk(c):
            by_paper[c["source"]].append(c)
    avoid = [t for f in ("multihop.json", "multihop_kg.json") for t in json.loads((DIR / f).read_text(encoding="utf-8"))]
    existing = json.loads(MULTI_RAW.read_text(encoding="utf-8")) if MULTI_RAW.exists() else []
    items = make_multihop(by_paper, n, MULTI_RAW, existing, avoid, id_prefix="h",
                          search_filter={"source": {"$in": papers}})
    print(f"{len(items)} câu multi-hop -> {MULTI_RAW}")


KEEP = [[1, "Giữ (1)", "y"], [2, "Giữ, cần sửa (2)", ""], [0, "Loại (0)", "n"]]
EDITS = RDIR / "edits.json"   # câu nhãn 2: id -> {"question", "ground_truth"} viết theo ghi chú của người duyệt
GUIDE = [
    ["Mỗi mục là một câu hỏi mới", "Đọc câu hỏi, đáp án chuẩn và đoạn gốc (multi-hop: hai đoạn của hai bài khác "
     "nhau), rồi chọn Giữ / Giữ, cần sửa / Loại. Thứ tự câu đã xáo trộn."],
    ["Giữ (1)", "Câu hỏi tự đứng được (nêu rõ tên phương pháp, hệ thống, khái niệm; không \"the passage\", \"this "
     "paper\", \"Table X\"), trả lời được từ đoạn gốc, và đáp án chuẩn đúng, đủ theo đoạn gốc."],
    ["Giữ, cần sửa (2)", "Câu dùng được nhưng cần sửa câu hỏi hoặc đáp án (mơ hồ, đáp án thiếu / thừa so với đoạn "
     "gốc, câu trả lời chỉ là một con số trơ trọi...). Ghi cách sửa vào ô ghi chú, ví dụ \"đáp án thêm: ...\" hoặc "
     "\"hỏi rõ là Kyber-768\"; câu sau khi sửa được đưa lại để bạn xem."],
    ["Loại (0)", "Câu hỏi về công thức lỗi ký tự, số hiệu bảng / hình, tên tác giả, chi tiết vô nghĩa; đáp án sai "
     "hoặc không có trong đoạn gốc; trùng ý với câu khác; multi-hop mà một đoạn đã trả lời đủ (không thật sự cần "
     "cả hai bài)."],
]


def load_raw() -> list[dict]:
    return [t for f in (SINGLE_RAW, MULTI_RAW) if f.exists() for t in json.loads(f.read_text(encoding="utf-8"))]


def make_sheet():
    raw = load_raw()
    rng = random.Random(config.SEED)
    order = rng.sample(raw, len(raw))
    cases, key = [], {}
    for n, t in enumerate(order, 1):
        code = f"Q{n:03d}"
        key[code] = t["id"]
        if t["type"] == "single":
            blocks = [{"title": "Bài", "text": t["source"]},
                      {"title": f"Đoạn gốc (trang {t['page']})", "text": t["chunk_text"]}]
        else:
            blocks = [b for i, s in enumerate(t["sources"]) for b in
                      ({"title": f"Bài {'AB'[i]}", "text": s}, {"title": f"Đoạn của bài {'AB'[i]}", "text": t["chunk_texts"][i]})]
        blocks = [{"title": "Câu hỏi", "text": t["question"]}, {"title": "Đáp án chuẩn", "text": t["ground_truth"]},
                  *blocks, {"key": "keep", "group": "Kết quả duyệt", "prompt": "Câu này:", "options": KEEP}]
        cases.append({"code": code, "title": "single-hop" if t["type"] == "single" else "multi-hop", "blocks": blocks})
    payload = {"title": "Duyệt câu hỏi giai đoạn 3", "file": LABELS.name, "guide": GUIDE, "cases": cases}
    payload["sheet_id"] = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()[:12]
    RDIR.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    SHEET.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", data, 1), encoding="utf-8")
    KEY.write_text(json.dumps({"seed": config.SEED, "sheet_id": payload["sheet_id"], "cases": key}, indent=1),
                   encoding="utf-8")
    print(f"{len(cases)} câu -> {SHEET}")


def apply_review(labels_path=None):
    from stage3_review import apply_review_files
    audit = apply_review_files([(SINGLE_RAW, SINGLE), (MULTI_RAW, MULTI)], KEY,
                               labels_path or LABELS, EDITS, config.RESULTS_DIR)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--multihop", type=int, default=0)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--labels", type=Path, help="file JSON nhãn người duyệt đã xuất")
    ap.add_argument("--limit", type=int, default=0, help="single-hop: chỉ N bài đầu (chạy thử)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if args.single:
        make_single(args.seed, args.limit)
    if args.multihop:
        make_multi(args.multihop, args.seed)
    if args.sheet:
        make_sheet()
    if args.apply:
        apply_review(args.labels)
    if not (args.single or args.multihop or args.sheet or args.apply):
        ap.print_help()


if __name__ == "__main__":
    main()
