"""Bước 3: chạy 2 hệ thống trên cùng bộ câu hỏi và chấm điểm bằng LLM-judge local.

    python evaluate.py                       # chỉ câu đã duyệt (reviewed=true, keep=true)
    python evaluate.py --include-unreviewed  # dùng cả câu chưa duyệt (chạy thử)
    python evaluate.py --limit 5             # chạy thử 5 câu

Có thể dừng giữa chừng (Ctrl+C) và chạy lại: kết quả đã có được giữ trong results/.

Định nghĩa chỉ số (theo RAGAS, Es et al. 2023):
  - faithfulness      = #claim được context hỗ trợ / #claim trong câu trả lời
  - answer_relevancy  = trung bình cosine(câu hỏi gốc, 3 câu hỏi sinh ngược từ câu trả lời);
                        = 0 nếu câu trả lời né tránh (noncommittal / từ chối)
  - hallucination     = câu trả lời có >= 1 claim KHÔNG được context hỗ trợ (0/1)
    hallucination_rate = tỉ lệ câu trả lời bị hallucination trên tổng số câu hỏi
    claim_halluc_rate  = #claim không được hỗ trợ / tổng #claim
Chỉ số phụ (để thấy Self-RAG không "ăn gian" bằng cách từ chối nhiều):
  - refusal_rate, answer_correctness (so với ground_truth), retrieval_hit (bài nguồn có trong context),
    latency_s, llm_calls
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

import config
from common import get_embeddings, get_llm, is_refusal

SYSTEMS = {
    "langchain_rag": "rag_langchain",
    "langgraph_selfrag": "rag_selfrag",
}

# ---------------- Judge prompts ----------------
EXTRACT_CLAIMS = ChatPromptTemplate.from_messages([
    ("system", "Break the answer into a list of short, atomic, self-contained factual statements "
               "(claims). Resolve pronouns. Ignore filler phrases. "
               'Return JSON: {{"claims": ["...", "..."]}}'),
    ("human", "Question: {question}\nAnswer: {answer}"),
])

VERIFY_CLAIMS = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact checker. For each numbered statement, decide whether it can be "
               "directly inferred from the context. Use 'yes' only if the context supports it; "
               "use 'no' if it is contradicted OR not mentioned. "
               'Return JSON: {{"verdicts": [{{"id": 1, "supported": "yes" or "no"}}, ...]}} '
               "with one entry per statement."),
    ("human", "Context:\n{context}\n\nStatements:\n{statements}"),
])

REVERSE_QUESTIONS = ChatPromptTemplate.from_messages([
    ("system", "Generate 3 different questions that the given answer would answer. "
               "Also say whether the answer is noncommittal (evasive, vague, 'I don't know', "
               "'cannot find'). "
               'Return JSON: {{"questions": ["...", "...", "..."], "noncommittal": true or false}}'),
    ("human", "Answer: {answer}"),
])

CORRECTNESS = ChatPromptTemplate.from_messages([
    ("system", "Compare a candidate answer with a reference answer to the question. "
               "Score 1 if the candidate conveys the key facts of the reference without contradicting "
               "it, otherwise 0. "
               'Return JSON: {{"score": 0 or 1}}'),
    ("human", "Question: {question}\nReference: {reference}\nCandidate: {answer}"),
])


def judge(prompt, **kw) -> dict:
    raw = (prompt | get_llm(config.JUDGE_MODEL, json_mode=True) | StrOutputParser()).invoke(kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score(item: dict, run: dict) -> dict:
    q, ans = item["question"], run["answer"]
    ctx = "\n\n".join(c["text"] for c in run["contexts"]) or "(no context)"
    refusal = is_refusal(ans)
    s = {"refusal": int(refusal),
         "retrieval_hit": int(any(c["source"] == item["source"] for c in run["contexts"]))}

    # Faithfulness + hallucination
    if refusal:
        s.update(n_claims=0, n_unsupported=0, faithfulness=math.nan, hallucination=0)
    else:
        claims = [c for c in judge(EXTRACT_CLAIMS, question=q, answer=ans).get("claims", [])
                  if isinstance(c, str) and c.strip()]
        if not claims:
            claims = [ans]
        stmts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        verdicts = judge(VERIFY_CLAIMS, context=ctx, statements=stmts).get("verdicts", [])
        sup = {int(v.get("id", 0)): str(v.get("supported", "")).lower().startswith("y")
               for v in verdicts if isinstance(v, dict) and str(v.get("id", "")).isdigit()}
        supported = sum(sup.get(i + 1, False) for i in range(len(claims)))  # thiếu verdict = không hỗ trợ
        s.update(n_claims=len(claims), n_unsupported=len(claims) - supported,
                 faithfulness=supported / len(claims),
                 hallucination=int(supported < len(claims)), claims=claims)

    # Answer relevancy
    if refusal:
        s["answer_relevancy"] = 0.0
    else:
        r = judge(REVERSE_QUESTIONS, answer=ans)
        gen_qs = [x for x in r.get("questions", []) if isinstance(x, str) and x.strip()][:3]
        if r.get("noncommittal") is True or not gen_qs:
            s["answer_relevancy"] = 0.0
        else:
            emb = get_embeddings()
            qv, gvs = emb.embed_query(q), emb.embed_documents(gen_qs)
            s["answer_relevancy"] = float(np.mean([cosine(qv, g) for g in gvs]))

    # Correctness vs ground truth
    s["answer_correctness"] = 0 if refusal else int(
        judge(CORRECTNESS, question=q, reference=item["ground_truth"], answer=ans).get("score", 0) in (1, "1"))
    return s


MAIN_NOTES = """\
- **faithfulness** (0–1, cao = tốt): câu trả lời được tách thành các claim nguyên tử; judge kiểm tra
  từng claim có suy ra được từ context đã truy xuất không. Điểm = #claim được hỗ trợ / #claim.
  Câu từ chối trả lời không có claim nên không tính vào trung bình.
- **answer_relevancy** (0–1, cao = tốt): judge sinh ngược 3 câu hỏi từ câu trả lời; điểm = trung bình
  cosine similarity (embedding) giữa câu hỏi gốc và 3 câu hỏi đó. Câu trả lời né tránh/từ chối = 0.
- **hallucination_rate** (0–1, thấp = tốt): tỉ lệ câu hỏi mà câu trả lời có ít nhất 1 claim
  không được context hỗ trợ, tính trên toàn bộ câu hỏi.
"""

EXTRA_NOTES = """\
- **claim_halluc_rate**: tỉ lệ claim sai trên tổng số claim. Cho biết mức độ hallucination
  (sai 1 chi tiết nhỏ hay sai toàn bộ), bổ sung cho hallucination_rate vốn chỉ đếm 0/1 theo câu.
- **halluc_rate_among_answered**: hallucination_rate chỉ tính trên các câu có trả lời (bỏ câu từ chối).
  Dùng để kiểm tra hệ thống có giảm hallucination thật hay chỉ nhờ từ chối nhiều hơn.
- **answer_correctness**: judge so câu trả lời với đáp án chuẩn (0/1). Đảm bảo việc giảm hallucination
  không làm giảm độ đúng của câu trả lời.
- **refusal_rate**: tỉ lệ câu hệ thống trả lời "không tìm thấy trong tài liệu". Self-RAG có cơ chế
  abstain nên tỉ lệ này thường cao hơn; cần đọc cùng answer_relevancy (câu từ chối = 0 điểm).
- **retrieval_hit**: tỉ lệ câu hỏi mà bài báo nguồn của câu hỏi có trong context. Cho biết lỗi đến từ
  khâu truy xuất hay khâu sinh câu trả lời.
- **latency_s / llm_calls**: thời gian và số lần gọi LLM trung bình mỗi câu, tức chi phí của
  vòng tự kiểm tra trong Self-RAG.
"""


# ---------------- IO helpers ----------------
def load_jsonl(path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f) if d}


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def bootstrap_ci(x, n=2000, seed=config.SEED):
    x = np.asarray([v for v in x if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(config.TESTSET_FILE, encoding="utf-8") as f:
        testset = [t for t in json.load(f)
                   if t.get("keep", True) and (args.include_unreviewed or t.get("reviewed"))]
    if args.limit:
        testset = testset[: args.limit]
    if not testset:
        raise SystemExit("Không có câu hỏi nào. Duyệt testset (reviewed=true) hoặc thêm --include-unreviewed.")
    print(f"{len(testset)} câu hỏi | LLM={config.LLM_MODEL} | judge={config.JUDGE_MODEL}")
    config.RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for sys_name in args.systems:
        module = __import__(SYSTEMS[sys_name])
        ans_path = config.RESULTS_DIR / f"answers_{sys_name}.jsonl"
        sc_path = config.RESULTS_DIR / f"scores_{sys_name}.jsonl"
        answers, scores = load_jsonl(ans_path), load_jsonl(sc_path)

        for item in tqdm(testset, desc=f"[{sys_name}] trả lời"):
            if item["id"] not in answers:
                answers[item["id"]] = {"id": item["id"], **module.answer(item["question"])}
                append_jsonl(ans_path, answers[item["id"]])

        for item in tqdm(testset, desc=f"[{sys_name}] chấm điểm"):
            if item["id"] not in scores:
                scores[item["id"]] = {"id": item["id"], **score(item, answers[item["id"]])}
                append_jsonl(sc_path, scores[item["id"]])

        for item in testset:
            a, s = answers[item["id"]], scores[item["id"]]
            rows.append({"system": sys_name, "id": item["id"], "topic": item["topic"],
                         "question": item["question"], "answer": a["answer"],
                         "latency_s": a["latency_s"], "llm_calls": a["llm_calls"],
                         **{k: v for k, v in s.items() if k not in ("id", "claims")}})

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "per_question.csv", index=False, encoding="utf-8-sig")

    metrics = ["faithfulness", "answer_relevancy", "hallucination", "answer_correctness",
               "refusal", "retrieval_hit", "latency_s", "llm_calls"]
    summary = []
    for sys_name, g in df.groupby("system", sort=False):
        row = {"system": sys_name, "n": len(g)}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].tolist())
            name = {"hallucination": "hallucination_rate", "refusal": "refusal_rate"}.get(m, m)
            row[name] = round(mean, 4)
            row[f"{name}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        row["claim_halluc_rate"] = round(g["n_unsupported"].sum() / max(g["n_claims"].sum(), 1), 4)
        answered = g[g["refusal"] == 0]
        row["halluc_rate_among_answered"] = round(answered["hallucination"].mean(), 4) if len(answered) else math.nan
        summary.append(row)
    sm = pd.DataFrame(summary)
    sm.to_csv(config.RESULTS_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    # Bảng chính: đúng 3 chỉ số của ý 1 (kèm CI 95%)
    main = ["faithfulness", "answer_relevancy", "hallucination_rate"]
    table = sm[["system", "n"] + [c for m in main for c in (m, f"{m}_ci95")]].to_markdown(index=False)
    # Chỉ số phụ (không thuộc yêu cầu ý 1, chỉ để giải thích kết quả)
    extra_cols = ["system", "claim_halluc_rate", "halluc_rate_among_answered", "answer_correctness",
                  "refusal_rate", "retrieval_hit", "latency_s", "llm_calls"]
    extra = sm[extra_cols].to_markdown(index=False)
    with open(config.RESULTS_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG\n\n"
                f"LLM: `{config.LLM_MODEL}` · Judge: `{config.JUDGE_MODEL}` · "
                f"Embedding: `{config.EMBED_MODEL}` · top-k={config.TOP_K}\n\n"
                f"## 1. Ba chỉ số chính\n\n{table}\n\n"
                f"*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*\n\n"
                f"{MAIN_NOTES}\n"
                f"## 2. Chỉ số phụ (giải thích kết quả)\n\n{extra}\n\n{EXTRA_NOTES}")
    print("\n" + table)
    print(f"\nĐã lưu: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
