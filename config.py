"""Cấu hình chung cho toàn bộ pipeline (ý 1: LangChain RAG vs LangGraph Self-RAG).

Mọi giá trị đều có thể ghi đè bằng biến môi trường cùng tên.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", PROJECT_DIR / "papers"))

# Dữ liệu đi kèm repo (clone về là chạy được, không cần build lại):
#   data/chunks.jsonl   văn bản 6.649 chunk          data/chroma.zip  vector DB Chroma (nén)
#   data/MANIFEST.json  sha256, tham số chunking, version thư viện / Ollama / model
# Lần chạy đầu, chroma.zip tự giải nén ra data/chroma/ (thư mục này không commit, xem .gitignore).
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_SNAPSHOT = DATA_DIR / "chroma.zip"
MANIFEST_FILE = DATA_DIR / "MANIFEST.json"
COLLECTION = os.getenv("COLLECTION", "papers")

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_DIR / "results"))
TESTSET_FILE = PROJECT_DIR / "testset" / "testset.json"        # single-hop
TESTSET_FILES = [TESTSET_FILE, PROJECT_DIR / "testset" / "multihop.json",
                 PROJECT_DIR / "testset" / "unanswerable.json"]

# ---- Server local (Ollama) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
# Judge dùng để chấm điểm: khác họ model với LLM_MODEL để tránh thiên lệch "tự chấm bài mình".
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
NUM_CTX = int(os.getenv("NUM_CTX", 4096))  # vừa VRAM 6GB

# ---- Chunking / retrieval (dùng chung cho cả 2 hệ thống để so sánh công bằng) ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
DROP_REFERENCES = True        # bỏ phần "References" cuối bài
DEDUP_IDENTICAL_FILES = True  # bỏ qua file PDF trùng byte-by-byte

# ---- Self-RAG ----
MAX_QUERY_REWRITES = int(os.getenv("MAX_QUERY_REWRITES", 2))
MAX_REGENERATIONS = int(os.getenv("MAX_REGENERATIONS", 2))

SEED = 42
