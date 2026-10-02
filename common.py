"""Thành phần dùng chung: LLM, embedding, vector store, prompt sinh câu trả lời.

Cả LangChain RAG và LangGraph Self-RAG dùng CHUNG retriever + prompt sinh câu trả lời
để khác biệt kết quả chỉ đến từ luồng điều khiển (self-reflection) của Self-RAG.
"""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config


class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text cần tiền tố task để đạt chất lượng tốt nhất."""

    def embed_documents(self, texts):
        if "nomic" in self.model:
            texts = [f"search_document: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        if "nomic" in self.model:
            text = f"search_query: {text}"
        return super().embed_query(text)


@lru_cache
def get_embeddings():
    return NomicEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


@lru_cache
def get_llm(model: str | None = None, json_mode: bool = False):
    return ChatOllama(
        model=model or config.LLM_MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0,
        seed=config.SEED,
        num_ctx=config.NUM_CTX,
        format="json" if json_mode else None,
    )


@lru_cache
def get_vectorstore():
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_retriever(k: int = config.TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source')}, page {d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


REFUSAL_TEXT = "I cannot find the answer in the provided documents."

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant answering questions about scientific papers. "
     "Answer ONLY using the context below. Be concise (at most 5 sentences). "
     f"If the context does not contain the answer, reply exactly: \"{REFUSAL_TEXT}\"\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def is_refusal(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith("i cannot find") or "cannot find the answer" in a or a == ""


def docs_to_contexts(docs: list[Document]) -> list[dict]:
    return [{"text": d.page_content, **d.metadata} for d in docs]
