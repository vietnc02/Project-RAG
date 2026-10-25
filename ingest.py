"""Bước 1: đọc toàn bộ PDF trong papers/ -> làm sạch -> chunk -> embed -> Chroma.

    python ingest.py               # build index từ PDF (bỏ qua nếu đã có)
    python ingest.py --reset       # xoá và build lại từ PDF
    python ingest.py --restore     # giải nén lại data/chroma.zip -> data/chroma/ (thường tự động)
    python ingest.py --from-chunks # embed lại từ data/chunks.jsonl (không đọc lại PDF)
    python ingest.py --snapshot    # đóng gói Chroma hiện tại -> data/chroma.zip + data/MANIFEST.json

Trong repo: data/chunks.jsonl (văn bản các chunk) + data/chroma.zip (vector DB) + data/MANIFEST.json
(số lượng, sha256, tham số chunking, version thư viện/Ollama/model). Clone về là chạy được: lần đầu gọi
Chroma, common.get_vectorstore() tự giải nén chroma.zip. Thư mục data/chroma/ không commit vì
chroma.sqlite3 bị ghi lại mỗi lần mở -> git luôn báo thay đổi.
"""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

import config
from common import env_info, get_vectorstore, restore_chroma, sha256

SNAPSHOT, MANIFEST = config.CHROMA_SNAPSHOT, config.MANIFEST_FILE

REF_HEADING = re.compile(r"^\s*(?:[IVX]+\.|\d+\.?)?\s*(references|bibliography)\s*$",
                         re.IGNORECASE | re.MULTILINE)


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)        # nối từ bị gạch nối xuống dòng
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path) -> list[Document]:
    with pymupdf.open(path) as pdf:
        pages = [p.get_text() for p in pdf]

    if config.DROP_REFERENCES:
        start = int(len(pages) * 0.4)
        for i in range(start, len(pages)):
            m = REF_HEADING.search(pages[i])
            if m:
                pages = pages[:i] + [pages[i][:m.start()]]
                break

    rel = path.relative_to(config.PAPERS_DIR)
    meta = {"source": path.stem, "topic": rel.parts[0], "path": str(rel)}
    return [Document(page_content=clean(t), metadata={**meta, "page": i + 1})
            for i, t in enumerate(pages) if len(clean(t)) > 50]


def chunks_from_pdfs() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,
                                              chunk_overlap=config.CHUNK_OVERLAP)
    pdfs = sorted(config.PAPERS_DIR.rglob("*.pdf"))
    seen, chunks, skipped, failed = {}, [], [], []
    for path in tqdm(pdfs, desc="Đọc PDF"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if config.DEDUP_IDENTICAL_FILES and digest in seen:
            skipped.append((path.name, seen[digest]))
            continue
        seen[digest] = path.name
        try:
            pages = load_pdf(path)
        except Exception as e:  # PDF hỏng/mã hoá
            failed.append((path.name, str(e)))
            continue
        for j, c in enumerate(splitter.split_documents(pages)):
            c.metadata["chunk_id"] = f"{digest[:12]}-{j}"
            chunks.append(c)

    print(f"{len(seen) - len(failed)} bài báo -> {len(chunks)} chunk")
    for a, b in skipped:
        print(f"  [bỏ qua - trùng hệt] {a}  ==  {b}")
    for a, e in failed:
        print(f"  [lỗi] {a}: {e}")

    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, **c.metadata}, ensure_ascii=False) + "\n")
    return chunks


def chunks_from_file() -> list[Document]:
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"{len(rows)} chunk từ {config.CHUNKS_FILE}")
    return [Document(page_content=r.pop("text"), metadata=r) for r in rows]


def snapshot():
    """Đóng gói Chroma + ghi manifest để đẩy lên GitHub cùng chunks.jsonl."""
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    n_index = get_vectorstore(auto_restore=False)._collection.count()
    get_vectorstore.cache_clear()  # đóng kết nối trước khi nén
    with zipfile.ZipFile(SNAPSHOT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(config.CHROMA_DIR.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(config.DATA_DIR))
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "papers": len({r["source"] for r in rows}), "chunks": len(rows), "chroma_vectors": n_index,
        "collection": config.COLLECTION, "distance": "cosine",
        "chunking": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                     "drop_references": config.DROP_REFERENCES,
                     "dedup_identical_files": config.DEDUP_IDENTICAL_FILES},
        "files": {"chunks.jsonl": sha256(config.CHUNKS_FILE), "chroma.zip": sha256(SNAPSHOT)},
        "environment": env_info(),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {SNAPSHOT} ({SNAPSHOT.stat().st_size / 1e6:.1f} MB) và {MANIFEST}")


def restore():
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    restore_chroma()
    print(f"Chroma: {get_vectorstore(auto_restore=False)._collection.count()} vector")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--reset", action="store_true", help="xoá Chroma và build lại từ PDF")
    g.add_argument("--from-chunks", action="store_true", help="xoá Chroma và embed lại từ chunks.jsonl")
    g.add_argument("--snapshot", action="store_true", help="đóng gói Chroma -> data/chroma.zip")
    g.add_argument("--restore", action="store_true", help="giải nén data/chroma.zip")
    args = ap.parse_args()

    if args.snapshot:
        return snapshot()
    if args.restore:
        return restore()
    if (args.reset or args.from_chunks) and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    vs = get_vectorstore(auto_restore=False)
    if vs._collection.count() > 0:
        print(f"Index đã có {vs._collection.count()} chunk. Dùng --reset / --from-chunks để build lại.")
        return

    chunks = chunks_from_file() if args.from_chunks else chunks_from_pdfs()
    batch = 128
    for i in tqdm(range(0, len(chunks), batch), desc="Embedding"):
        part = chunks[i:i + batch]
        vs.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
    print(f"Xong. Index: {vs._collection.count()} chunk tại {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
