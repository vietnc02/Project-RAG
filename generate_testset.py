"""Bước 2: sinh bộ câu hỏi đánh giá từ các chunk (rồi DUYỆT TAY trước khi chạy evaluate).

    python generate_testset.py --per-paper 1

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


def good_chunk(c: dict) -> bool:
    t = c["text"]
    if len(t) < 600:
        return False
    letters = sum(ch.isalpha() for ch in t) / len(t)
    return letters > 0.65 and not re.search(r"acknowledg|copyright ©|all rights reserved", t, re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-paper", type=int, default=1)
    ap.add_argument("--max", type=int, default=0, help="giới hạn tổng số câu (0 = không)")
    args = ap.parse_args()
    random.seed(config.SEED)

    by_paper = defaultdict(list)
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if good_chunk(c):
                by_paper[c["source"]].append(c)

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
            except json.JSONDecodeError:
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
