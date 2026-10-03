"""Bước 2: sinh bộ câu hỏi đánh giá từ các chunk (rồi DUYỆT TAY trước khi chạy evaluate).

    python generate_testset.py --per-paper 1     # single-hop -> testset/testset.json
    python generate_testset.py --multihop 60     # multi-hop  -> testset/multihop_raw.json
    python generate_testset.py --multihop 40 --append --seed 7   # sinh thêm, không lặp cặp bài đã có

Multi-hop: ghép 1 chunk của bài A với chunk gần nghĩa nhất của một bài B khác, yêu cầu LLM đặt câu hỏi
cần thông tin của CẢ HAI bài; sau đó LLM kiểm tra lại từng đoạn riêng lẻ, chỉ giữ câu mà không đoạn nào
một mình trả lời đủ. Câu hỏi ngoài corpus (testset/unanswerable.json) được viết tay, không sinh tự động.

Ra file testset/testset.json. Mỗi mục có "reviewed": false. Khi duyệt:
  - sửa question/ground_truth nếu cần, đặt "reviewed": true
  - câu hỏi kém (mơ hồ, "this paper...", hỏi về bảng số liệu vô nghĩa) -> xoá hoặc đặt "keep": false
"""
import argparse
import json
import random
import re
from collections import defaultdict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ollama import ResponseError
from tqdm import tqdm

import config
from common import get_llm

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You create evaluation questions for a retrieval-augmented QA system over scientific papers.\n"
     "Given a passage, write ONE factual question that can be answered from the passage alone, "
     "and its reference answer (1-3 sentences, taken from the passage).\n"
     "Rules:\n"
     "- The question must be self-contained and specific: mention the concrete method, system, "
     "attack or concept by name. NEVER say 'this paper', 'the authors', 'the passage', 'Table X'.\n"
     "- Do not ask about citation numbers, figure numbers, or author names.\n"
     "- If the passage has no meaningful technical content (references, acknowledgements, "
     "garbled math), return {{\"skip\": true}}.\n"
     'Return JSON: {{"question": "...", "ground_truth": "..."}}'),
    ("human", "Paper title: {title}\n\nPassage:\n{text}"),
])

MULTIHOP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You create evaluation questions for a retrieval-augmented QA system over scientific papers.\n"
     "Given two passages from two DIFFERENT papers, write ONE question that can only be answered by "
     "combining information from BOTH passages (e.g. compare two methods/results, or relate a concept "
     "in one passage to a system in the other), and its reference answer (2-4 sentences) that uses facts "
     "from both passages.\n"
     "Rules:\n"
     "- The question must be self-contained and specific: name the concrete methods, systems, attacks "
     "or concepts. NEVER say 'passage', 'paper 1', 'this paper', 'the authors', 'Table X'.\n"
     "- Do not ask about citation numbers, figure numbers, or author names.\n"
     "- If the passages have no meaningful connection, return {{\"skip\": true}}.\n"
     'Return JSON: {{"question": "...", "ground_truth": "..."}}'),
    ("human", "Paper A: {title_a}\nPassage A:\n{text_a}\n\nPaper B: {title_b}\nPassage B:\n{text_b}"),
])

ALONE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Can the question be answered COMPLETELY using only the passage below? "
               'Return JSON: {{"complete": "yes" or "no"}}'),
    ("human", "Passage:\n{text}\n\nQuestion: {question}"),
])


def good_chunk(c: dict) -> bool:
    t = c["text"]
    if len(t) < 600:
        return False
    letters = sum(ch.isalpha() for ch in t) / len(t)
    return letters > 0.65 and not re.search(r"acknowledg|copyright ©|all rights reserved", t, re.I)


def make_multihop(by_paper: dict, n: int, out, existing: list[dict] = ()) -> list[dict]:
    """Sinh thêm n câu; `existing` = câu đã có (lượt trước) -> không dùng lại cặp bài / chunk đã dùng."""
    from common import get_vectorstore
    vs = get_vectorstore()
    gen = MULTIHOP_PROMPT | get_llm(json_mode=True) | StrOutputParser()
    alone = ALONE_PROMPT | get_llm(json_mode=True) | StrOutputParser()
    items, target = list(existing), len(existing) + n
    used_b = defaultdict(int)
    used_pairs = {frozenset(t["sources"]) for t in items}
    used_chunks = {c for t in items for c in t["chunk_ids"]}
    for t in items:
        used_b[t["sources"][1]] += 1
    for title in tqdm(random.sample(sorted(by_paper), len(by_paper)), desc="Sinh câu multi-hop"):
        if len(items) >= target:
            break
        free = [c for c in by_paper[title] if c["chunk_id"] not in used_chunks]
        if not free:
            continue
        a = random.choice(free)
        # chunk gần nghĩa nhất thuộc bài khác (mỗi bài làm B tối đa 2 lần, không lặp cặp bài đã có)
        cand = [d for d in vs.similarity_search(a["text"], k=30)
                if d.metadata["source"] != title and used_b[d.metadata["source"]] < 2
                and frozenset((title, d.metadata["source"])) not in used_pairs
                and d.metadata["chunk_id"] not in used_chunks and good_chunk({"text": d.page_content})]
        if not cand:
            continue
        b = cand[0]
        try:
            d = json.loads(gen.invoke({"title_a": title, "text_a": a["text"],
                                       "title_b": b.metadata["source"], "text_b": b.page_content}))
        except (json.JSONDecodeError, ResponseError):  # JSON hỏng / Ollama huỷ do lặp token
            continue
        if d.get("skip") or not d.get("question") or not d.get("ground_truth"):
            continue
        if re.search(r"\bpassage\b|\bpaper [ab12]\b|\bthis (paper|study|work)\b|\bthe authors\b", d["question"], re.I):
            continue
        # bỏ câu mà một đoạn đơn lẻ đã trả lời đủ (không thực sự multi-hop)
        complete = []
        for text in (a["text"], b.page_content):
            try:
                r = json.loads(alone.invoke({"text": text, "question": d["question"]}))
                complete.append(str(r.get("complete", "")).lower().startswith("y"))
            except (json.JSONDecodeError, ResponseError):  # JSON hỏng / Ollama huỷ do lặp token
                complete.append(True)
        if any(complete):
            continue
        used_b[b.metadata["source"]] += 1
        used_pairs.add(frozenset((title, b.metadata["source"])))
        items.append({
            "id": f"m{len(items) + 1:03d}", "type": "multihop",
            "question": d["question"].strip(), "ground_truth": d["ground_truth"].strip(),
            "sources": [title, b.metadata["source"]], "topic": a["topic"],
            "chunk_ids": [a["chunk_id"], b.metadata["chunk_id"]], "chunk_texts": [a["text"], b.page_content],
            "reviewed": False, "keep": True,
        })
        out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")  # lưu dần
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-paper", type=int, default=1)
    ap.add_argument("--max", type=int, default=0, help="giới hạn tổng số câu (0 = không)")
    ap.add_argument("--multihop", type=int, default=0, help="sinh N câu multi-hop -> testset/multihop_raw.json")
    ap.add_argument("--append", action="store_true", help="multi-hop: sinh thêm vào multihop_raw.json đã có")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    random.seed(args.seed)

    by_paper = defaultdict(list)
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if good_chunk(c):
                by_paper[c["source"]].append(c)

    if args.multihop:
        out = config.TESTSET_FILE.parent / "multihop_raw.json"
        existing = json.loads(out.read_text(encoding="utf-8")) if args.append and out.exists() else []
        items = make_multihop(by_paper, args.multihop, out, existing)
        print(f"Đã sinh {len(items)} câu multi-hop -> {out}. Duyệt tay rồi lưu thành testset/multihop.json")
        return

    chain = PROMPT | get_llm(json_mode=True) | StrOutputParser()
    items = []
    papers = sorted(by_paper)
    for title in tqdm(papers, desc="Sinh câu hỏi"):
        cands = random.sample(by_paper[title], len(by_paper[title]))
        made = 0
        for c in cands[: args.per_paper * 3]:  # thử tối đa 3 chunk / câu cần sinh
            if made >= args.per_paper:
                break
            try:
                d = json.loads(chain.invoke({"title": title, "text": c["text"]}))
            except (json.JSONDecodeError, ResponseError):  # JSON hỏng / Ollama huỷ do lặp token
                continue
            if d.get("skip") or not d.get("question") or not d.get("ground_truth"):
                continue
            if re.search(r"\bthis (paper|study|passage|work)\b|\bthe authors\b", d["question"], re.I):
                continue
            items.append({
                "id": f"q{len(items) + 1:03d}",
                "question": d["question"].strip(),
                "ground_truth": d["ground_truth"].strip(),
                "source": title, "topic": c["topic"], "page": c["page"],
                "chunk_id": c["chunk_id"], "chunk_text": c["text"],
                "reviewed": False, "keep": True,
            })
            made += 1
        if args.max and len(items) >= args.max:
            break

    config.TESTSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TESTSET_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Đã sinh {len(items)} câu hỏi từ {len(papers)} bài -> {config.TESTSET_FILE}")
    print("Hãy duyệt tay file này (sửa / đặt keep=false / reviewed=true) trước khi chạy evaluate.py")


if __name__ == "__main__":
    main()
