"""Baseline không truy xuất (closed-book): LLM sinh câu trả lời trả lời từ kiến thức sẵn có, không có context.

    python baseline_closed_book.py            # mọi câu hỏi của ý 1 và ý 2-4 -> results/<RESULTS_DIR>/closed_book/
    python baseline_closed_book.py --limit 3  # chạy thử

Mục đích (bản đánh giá, X2): tách phần câu trả lời đúng nhờ context khỏi phần mô hình 7B đã biết sẵn. Cùng LLM
(qwen2.5:7b, temperature 0, seed cố định), cùng giới hạn độ dài câu trả lời và cùng câu refuse như prompt RAG, chỉ
bỏ context và cho phép trả lời từ kiến thức riêng. Chấm answer correctness bằng đúng prompt CORRECTNESS của ý 1–2
(judge llama3.1:8b); câu out-of-corpus: đúng = refuse. Không tính faithfulness (không có context).
Câu hỏi = hợp của bộ câu hỏi ý 1 (144 câu) và ý 2–4 (85 câu); câu trùng id chỉ chạy một lần.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

import config
from common import REFUSAL_TEXT, env_info, get_llm, is_refusal
from evaluate import append_jsonl, flush_judge_log, judge_log_md, load_jsonl, load_testset
from evaluate_graphrag import TESTSET_FILES as Y2_FILES, answer_correctness

CLOSED_BOOK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant answering questions about scientific papers. No documents are provided: answer from "
     "your own knowledge. Be concise (at most 5 sentences). "
     f"If you do not know the answer, reply exactly: \"{REFUSAL_TEXT}\""),
    ("human", "{question}"),
])


def _files():
    files = []
    for f in [*config.TESTSET_FILES, *Y2_FILES]:
        if f.exists() and f not in files:
            files.append(f)
    return files


def load_items() -> list[dict]:
    """Hợp các file câu hỏi (không trùng file), bỏ câu trùng id."""
    seen, out = set(), []
    for f in _files():
        for t in load_testset([f]):
            if t["id"] not in seen:
                seen.add(t["id"])
                out.append(t)
    return out


def answer(question: str) -> dict:
    t0 = time.perf_counter()
    out = (CLOSED_BOOK_PROMPT | get_llm() | StrOutputParser()).invoke({"question": question})
    return {"answer": out.strip(), "latency_s": round(time.perf_counter() - t0, 2), "llm_calls": 1,
            "trace": ["generate (closed-book)"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    items = load_items()[: args.limit or None]
    out = config.RESULTS_DIR / ("closed_book" + ("_thu" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    info = {"started": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv[1:],
            "prompt": CLOSED_BOOK_PROMPT.messages[0].prompt.template,
            "testset": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in _files()},
            "n_items": len(items), "environment": env_info()}
    ri = out / "run_info.json"
    if ri.exists():
        info = {**json.loads(ri.read_text(encoding="utf-8")),
                f"judge_pass_{config.JUDGE_PASS}": {"started": info["started"], "environment": info["environment"]}}
    ri.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    ans_path, sc_path = out / "answers.jsonl", config.pass_path(out / "scores.jsonl")
    answers, scores = load_jsonl(ans_path), load_jsonl(sc_path)
    for t in tqdm(items, desc="closed-book: trả lời"):
        if t["id"] not in answers:
            answers[t["id"]] = {"id": t["id"], **answer(t["question"])}
            append_jsonl(ans_path, answers[t["id"]])
    for t in tqdm(items, desc=f"closed-book: chấm (lượt {config.JUDGE_PASS})"):
        if t["id"] not in scores:
            a = answers[t["id"]]["answer"]
            ref = is_refusal(a)
            scores[t["id"]] = {"id": t["id"], "refusal": int(ref), "answer_correctness": answer_correctness(t, a, ref)}
            append_jsonl(sc_path, scores[t["id"]])
            flush_judge_log(config.pass_path(out / "judge_log.jsonl"), t["id"])
    by = {}
    for t in items:
        s = scores[t["id"]]
        g = by.setdefault(t["type"], {"n": 0, "correct": 0, "refused": 0})
        g["n"] += 1
        g["correct"] += s["answer_correctness"]
        g["refused"] += s["refusal"]
    lines = ["# Baseline không truy xuất (closed-book)\n",
             f"LLM `{config.LLM_MODEL}` không có context; judge `{config.JUDGE_MODEL}` (prompt CORRECTNESS của ý 1–2).\n",
             "| loại câu | n | answer correctness | refusal |", "|---|---|---|---|",
             *(f"| {k} | {v['n']} | {v['correct'] / v['n']:.3f} | {v['refused'] / v['n']:.3f} |" for k, v in by.items()),
             "", *judge_log_md(out)]
    (out / f"summary.p{config.JUDGE_PASS}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
