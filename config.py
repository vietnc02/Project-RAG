"""Cấu hình chung cho toàn bộ pipeline (ý 1: LangChain RAG vs LangGraph Self-RAG; ý 2: GraphRAG vs Vector RAG).

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

RESULTS_DIR = PROJECT_DIR / os.getenv("RESULTS_DIR", "results")   # đường dẫn tương đối tính từ thư mục dự án
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

# ---- Ý 2-3: Knowledge Graph / GraphRAG (30 bài chủ đề quantum) ----
# 23 bài Quantum Security + Quantum ML trong papers/ + 7 bài bổ sung trong papers_y2/ (tách riêng: dữ liệu ý 1 không đổi)
# data/kg/papers.json      : 30 bài, xếp theo thứ tự sao cho k bài đầu (k = 10, 20, 30) giữ tỉ lệ chủ đề
# data/kg/chunks_y2.jsonl  : chunk của 7 bài bổ sung (cùng cách chia chunk với ý 1)
# data/kg/chroma.zip       : Chroma riêng của ý 2 (collection kg30 = chunk của đúng 30 bài), tự giải nén ra data/kg/chroma/
# data/kg/extractions.jsonl: entity/relation LLM trích từ từng chunk (cache, dùng chung cho mọi k)
# data/kg/k<k>/            : KG đã gộp (graph.json) + vector entity/relation (vectors.npz)
PAPERS_Y2_DIR = PROJECT_DIR / "papers_y2"
KG_DIR = DATA_DIR / "kg"
KG_PAPERS_FILE = KG_DIR / "papers.json"
KG_CHUNKS_Y2 = KG_DIR / "chunks_y2.jsonl"
KG_CHROMA_DIR = KG_DIR / "chroma"
KG_CHROMA_SNAPSHOT = KG_DIR / "chroma.zip"
KG_MANIFEST = KG_DIR / "MANIFEST.json"
KG_COLLECTION = "kg30"
KG_EXTRACTIONS = KG_DIR / "extractions.jsonl"
KG_K = int(os.getenv("KG_K", 30))           # số bài dùng cho KG / GraphRAG
# Bước trích xuất KG: đưa toàn bộ layer của LLM lên GPU. Mặc định Ollama ước tính dư bộ nhớ và đẩy ~18% model
# sang CPU trên GPU 6GB (chậm hơn 1,5 lần dù VRAM còn trống). Đặt -1 để Ollama tự chia (máy ít VRAM hơn).
KG_NUM_GPU = int(os.getenv("KG_NUM_GPU", 99))
KG_SEED_ENTITIES = int(os.getenv("KG_SEED_ENTITIES", 8))   # số entity khớp với câu hỏi
KG_MAX_RELATIONS = int(os.getenv("KG_MAX_RELATIONS", 12))  # số quan hệ đưa vào context

SEED = 42

# ---- Chấm nhiều lượt, thử lại khi judge lỗi, điều kiện nhiễu thuần, Vector RAG theo k ----
# JUDGE_PASS: lượt chấm (0, 1, 2): seed của judge = SEED + 1000 * JUDGE_PASS; mỗi lượt ghi scores_*.p<N>.jsonl,
#             judge_passes.py gộp các lượt (nhãn nhị phân: đa số; điểm liên tục: trung bình) thành scores_*.jsonl.
# JUDGE_RETRIES: số lần thử lại khi judge trả JSON hỏng / Ollama huỷ / thiếu verdict (đổi seed mỗi lần thử);
#             verdict vẫn thiếu sau khi thử lại thì bị loại khỏi mẫu số (không còn tính là "không").
# NOISE_CHUNKS: thêm n đoạn ngẫu nhiên của bài khác vào top-k (điều kiện nhiễu thuần của ý 1).
# SOURCE_FILTER: chỉ truy xuất trong các bài này (Vector RAG theo k bài của ý 3); đặt trong code.
JUDGE_PASS = int(os.getenv("JUDGE_PASS", 0))
JUDGE_RETRIES = int(os.getenv("JUDGE_RETRIES", 3))
NOISE_CHUNKS = int(os.getenv("NOISE_CHUNKS", 0))
SOURCE_FILTER: list[str] | None = None


def pass_path(path: Path) -> Path:
    """scores_x.jsonl -> scores_x.p<JUDGE_PASS>.jsonl (file của một lượt chấm)."""
    return path.with_name(f"{path.stem}.p{JUDGE_PASS}{path.suffix}")
