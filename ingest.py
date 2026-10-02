"""Bước 1: đọc toàn bộ PDF trong papers/ -> làm sạch -> chunk -> embed -> Chroma.

    python ingest.py            # build index (bỏ qua nếu đã có)
    python ingest.py --reset    # xoá và build lại
"""
import argparse
import hashlib
import json
import re
import shutil

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

import config
from common import get_vectorstore

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset and config.DATA_DIR.exists():
        shutil.rmtree(config.DATA_DIR)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    vs = get_vectorstore()
    if vs._collection.count() > 0:
        print(f"Index đã có {vs._collection.count()} chunk. Dùng --reset để build lại.")
        return

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

    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, **c.metadata}, ensure_ascii=False) + "\n")

    batch = 128
    for i in tqdm(range(0, len(chunks), batch), desc="Embedding"):
        part = chunks[i:i + batch]
        vs.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
    print(f"Xong. Index: {vs._collection.count()} chunk tại {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
