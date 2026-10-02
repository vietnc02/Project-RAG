"""Hệ thống B: Self-RAG bằng LangGraph.

Luồng (dựa trên Asai et al., 2023 "Self-RAG" + LangGraph self-RAG pattern):

    retrieve -> grade_documents --(có tài liệu liên quan)--> generate -> grade_generation
                     |                                                     |
             (không có, còn lượt)                         grounded & useful -> END
                     v                                    không grounded   -> generate (bản chặt hơn)
              transform_query -> retrieve                 không useful     -> transform_query
                     |
             (hết lượt) -> abstain (từ chối trả lời)

    python rag_selfrag.py "What is a network tarpit?"
"""
import json
import sys
import time
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

import config
from common import (GENERATE_PROMPT, REFUSAL_TEXT, docs_to_contexts, format_docs, get_llm,
                    get_retriever, is_refusal)


class State(TypedDict, total=False):
    question: str          # câu hỏi gốc
    query: str             # truy vấn hiện tại (có thể đã được viết lại)
    documents: list[Document]
    generation: str
    feedback: str          # lý do lần sinh trước bị loại
    rewrites: int
    regenerations: int
    llm_calls: int
    trace: list[str]
    verdict: str           # kết quả grade_generation


# ---------------- Prompts ----------------
GRADE_DOC = ChatPromptTemplate.from_messages([
    ("system", "You assess whether a retrieved document is relevant to a user question. "
               "It is relevant if it contains keywords or information useful to answer the question. "
               'Respond with JSON: {{"score": "yes"}} or {{"score": "no"}}.'),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])

GRADE_HALLUCINATION = ChatPromptTemplate.from_messages([
    ("system", "You assess whether an answer is fully grounded in / supported by a set of facts. "
               "Answer 'yes' only if every statement in the answer is supported by the facts. "
               'Respond with JSON: {{"score": "yes" or "no", "reason": "<short reason>"}}.'),
    ("human", "Facts:\n{documents}\n\nAnswer: {generation}"),
])

GRADE_ANSWER = ChatPromptTemplate.from_messages([
    ("system", "You assess whether an answer addresses / resolves the question. "
               'Respond with JSON: {{"score": "yes"}} or {{"score": "no"}}.'),
    ("human", "Question: {question}\n\nAnswer: {generation}"),
])

REWRITE = ChatPromptTemplate.from_messages([
    ("system", "You rewrite a question into a better standalone search query for a vector store "
               "of scientific papers. Keep the original intent; add key technical terms. "
               "Output only the rewritten query."),
    ("human", "Original question: {question}\nPrevious query: {query}"),
])

STRICT_GENERATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant answering questions about scientific papers. "
     "Your previous answer was rejected because it contained statements NOT supported by the context "
     "({feedback}). Write a new answer using ONLY facts explicitly stated in the context. "
     "Be concise (at most 5 sentences). "
     f"If the context does not contain the answer, reply exactly: \"{REFUSAL_TEXT}\"\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def _yes(prompt, **kw) -> tuple[bool, str]:
    raw = (prompt | get_llm(json_mode=True) | StrOutputParser()).invoke(kw)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = {"score": raw}
    return str(d.get("score", "")).strip().lower().startswith("y"), str(d.get("reason", ""))


def _step(state: State, name: str, calls: int = 0) -> dict:
    return {"trace": state.get("trace", []) + [name],
            "llm_calls": state.get("llm_calls", 0) + calls}


# ---------------- Nodes ----------------
def retrieve(state: State) -> dict:
    query = state.get("query") or state["question"]
    return {"documents": get_retriever().invoke(query), "query": query, **_step(state, "retrieve")}


def grade_documents(state: State) -> dict:
    kept = [d for d in state["documents"]
            if _yes(GRADE_DOC, document=d.page_content, question=state["question"])[0]]
    return {"documents": kept,
            **_step(state, f"grade_documents({len(kept)}/{len(state['documents'])})",
                    len(state["documents"]))}


def generate(state: State) -> dict:
    ctx = {"context": format_docs(state["documents"]), "question": state["question"]}
    if state.get("feedback"):
        prompt, ctx["feedback"] = STRICT_GENERATE, state["feedback"]
    else:
        prompt = GENERATE_PROMPT
    gen = (prompt | get_llm() | StrOutputParser()).invoke(ctx).strip()
    return {"generation": gen, **_step(state, "generate", 1)}


def transform_query(state: State) -> dict:
    q = (REWRITE | get_llm() | StrOutputParser()).invoke(
        {"question": state["question"], "query": state.get("query", state["question"])})
    return {"query": q.strip().strip('"'), "rewrites": state.get("rewrites", 0) + 1,
            "feedback": "", **_step(state, "transform_query", 1)}


def abstain(state: State) -> dict:
    return {"generation": REFUSAL_TEXT, "documents": state.get("documents", []),
            **_step(state, "abstain")}


def grade_generation(state: State) -> dict:
    """Self-reflection: kiểm tra grounded (ISSUP) rồi useful (ISUSE)."""
    if is_refusal(state["generation"]):
        return {"verdict": "refusal", **_step(state, "grade_generation(refusal)")}
    grounded, reason = _yes(GRADE_HALLUCINATION, documents=format_docs(state["documents"]),
                            generation=state["generation"])
    if not grounded:
        return {"verdict": "not_grounded", "feedback": reason or "unsupported claims",
                **_step(state, "grade_generation(not_grounded)", 1)}
    useful, _ = _yes(GRADE_ANSWER, question=state["question"], generation=state["generation"])
    return {"verdict": "useful" if useful else "not_useful",
            **_step(state, f"grade_generation({'useful' if useful else 'not_useful'})", 2)}


# ---------------- Edges ----------------
def route_after_grading(state: State) -> str:
    if state["documents"]:
        return "generate"
    return "transform_query" if state.get("rewrites", 0) < config.MAX_QUERY_REWRITES else "abstain"


def route_after_generation(state: State) -> str:
    v = state["verdict"]
    if v in ("useful", "refusal"):
        return END
    if v == "not_grounded":
        return "regenerate" if state.get("regenerations", 0) < config.MAX_REGENERATIONS else "abstain"
    # grounded nhưng không trả lời đúng trọng tâm -> thử truy vấn khác, hết lượt thì giữ câu trả lời
    return "transform_query" if state.get("rewrites", 0) < config.MAX_QUERY_REWRITES else END


def regenerate(state: State) -> dict:
    return {"regenerations": state.get("regenerations", 0) + 1}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("retrieve", retrieve), ("grade_documents", grade_documents),
                     ("generate", generate), ("grade_generation", grade_generation),
                     ("transform_query", transform_query), ("regenerate", regenerate),
                     ("abstain", abstain)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            ["generate", "transform_query", "abstain"])
    g.add_edge("generate", "grade_generation")
    g.add_conditional_edges("grade_generation", route_after_generation,
                            ["regenerate", "transform_query", "abstain", END])
    g.add_edge("regenerate", "generate")
    g.add_edge("transform_query", "retrieve")
    g.add_edge("abstain", END)
    return g.compile()


_graph = None


def answer(question: str) -> dict:
    global _graph
    _graph = _graph or build_graph()
    t0 = time.perf_counter()
    out = _graph.invoke({"question": question, "rewrites": 0, "regenerations": 0,
                         "llm_calls": 0, "trace": []}, {"recursion_limit": 60})
    return {
        "answer": out["generation"],
        "contexts": docs_to_contexts(out.get("documents", [])),
        "latency_s": round(time.perf_counter() - t0, 2),
        "llm_calls": out["llm_calls"],
        "trace": out["trace"],
        "rewrites": out.get("rewrites", 0),
        "regenerations": out.get("regenerations", 0),
    }


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]) or "What is a network tarpit?")
    print(r["answer"])
    print("\nTrace:", " -> ".join(r["trace"]))
    print("Nguồn:", sorted({c["source"] for c in r["contexts"]}))
