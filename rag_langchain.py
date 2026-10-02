"""Hệ thống A: LangChain RAG thuần (retrieve top-k -> generate, 1 lần gọi LLM).

    python rag_langchain.py "What is a network tarpit?"
"""
import sys
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from common import GENERATE_PROMPT, docs_to_contexts, format_docs, get_llm, get_retriever


def build_chain():
    generate = GENERATE_PROMPT | get_llm() | StrOutputParser()
    return (
        RunnableParallel(docs=get_retriever(), question=RunnablePassthrough())
        | RunnablePassthrough.assign(
            answer=RunnableLambda(lambda x: {"context": format_docs(x["docs"]),
                                             "question": x["question"]}) | generate)
    )


_chain = None


def answer(question: str) -> dict:
    global _chain
    _chain = _chain or build_chain()
    t0 = time.perf_counter()
    out = _chain.invoke(question)
    return {
        "answer": out["answer"].strip(),
        "contexts": docs_to_contexts(out["docs"]),
        "latency_s": round(time.perf_counter() - t0, 2),
        "llm_calls": 1,
        "trace": ["retrieve", "generate"],
    }


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]) or "What is a network tarpit?")
    print(r["answer"])
    print("\nNguồn:", sorted({c["source"] for c in r["contexts"]}))
