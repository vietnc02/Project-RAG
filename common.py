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
        # JSON mode đôi khi lặp vô hạn (không tự dừng); đầu ra hợp lệ của các bước chấm chỉ vài trăm token
        num_predict=1024 if json_mode else None,
    )


def sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def restore_chroma():
    """Giải nén data/chroma.zip -> data/chroma/ (kiểm tra sha256 với MANIFEST.json)."""
    import json
    import zipfile
    want = json.loads(config.MANIFEST_FILE.read_text(encoding="utf-8"))["files"]["chroma.zip"]
    if sha256(config.CHROMA_SNAPSHOT) != want:
        raise SystemExit(f"{config.CHROMA_SNAPSHOT} không khớp sha256 trong {config.MANIFEST_FILE}")
    with zipfile.ZipFile(config.CHROMA_SNAPSHOT) as z:
        z.extractall(config.DATA_DIR)
    print(f"Đã giải nén Chroma -> {config.CHROMA_DIR}")


@lru_cache
def get_vectorstore(auto_restore: bool = True):
    if auto_restore and not config.CHROMA_DIR.exists() and config.CHROMA_SNAPSHOT.exists():
        restore_chroma()  # lần chạy đầu sau khi clone repo
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_retriever(k: int | None = None):
    # đọc config.TOP_K lúc gọi (không phải lúc import) để evaluate.py --top-k ghi đè được
    return get_vectorstore().as_retriever(search_kwargs={"k": k or config.TOP_K})


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


def env_info() -> dict:
    """Version thư viện + Ollama + digest model, ghi kèm mỗi lần chạy để tái lập kết quả."""
    import json
    import platform
    import urllib.request
    from importlib.metadata import PackageNotFoundError, version

    pkgs = {}
    for name in ("langchain", "langchain-core", "langgraph", "langchain-ollama", "langchain-chroma",
                 "langchain-text-splitters", "chromadb", "ollama", "pymupdf"):
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None

    def get(path):
        try:
            with urllib.request.urlopen(config.OLLAMA_URL + path, timeout=5) as r:
                return json.load(r)
        except Exception:
            return {}

    tags = {m["name"]: m for m in get("/api/tags").get("models", [])}
    models = {}
    for role, name in (("llm", config.LLM_MODEL), ("judge", config.JUDGE_MODEL), ("embedding", config.EMBED_MODEL)):
        m = tags.get(name) or tags.get(f"{name}:latest") or {}
        models[role] = {"name": name, "digest": m.get("digest", "")[:12],
                        "quantization": m.get("details", {}).get("quantization_level")}
    return {"python": platform.python_version(), "packages": pkgs,
            "ollama_server": get("/api/version").get("version"), "models": models}
