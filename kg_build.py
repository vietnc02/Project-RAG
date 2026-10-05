"""Ý 2-3: dựng Knowledge Graph (KG) từ k bài báo chủ đề quantum bằng LLM local.

    python kg_build.py --ingest            # chia chunk 7 bài bổ sung (papers_y2/) -> data/kg/chunks_y2.jsonl
    python kg_build.py --select            # chọn 30 bài -> data/kg/papers.json
    python kg_build.py --store             # Chroma riêng của ý 2 (30 bài) -> data/kg/chroma.zip + MANIFEST.json
    python kg_build.py --extract           # LLM trích entity/relation từng chunk -> data/kg/extractions.jsonl
    python kg_build.py --extract --k 10    #   chỉ chunk của 10 bài đầu (chạy tiếp được nếu dừng giữa chừng)
    python kg_build.py --build --k 30      # gộp entity/relation thành KG -> data/kg/k30/graph.json + vectors.npz
(các file trên đã có sẵn trong repo; clone về chỉ cần chạy evaluate_graphrag.py / rag_graphrag.py)

30 bài = 23 bài Quantum Security + Quantum Machine Learning có trong papers/ + 7 bài bổ sung trong papers_y2/
(xem papers_y2/SOURCES.md). Dữ liệu của ý 1 (papers/, data/chunks.jsonl, data/chroma.zip) không bị sửa: chunk của
7 bài mới nằm trong data/kg/chunks_y2.jsonl và Vector RAG của ý 2 dùng Chroma riêng (collection kg30) chứa đúng
chunk của 30 bài (23 bài cũ chép nguyên vector từ Chroma của ý 1, 7 bài mới embed bằng cùng model).

Cách dựng KG (theo hướng GraphRAG / LightRAG, tự cài đặt để kiểm soát từng bước):
  1. Trích xuất: mỗi chunk (đúng các chunk của Vector RAG) -> 1 lần gọi LLM, ra danh sách entity (tên, loại,
     mô tả) và relation (entity nguồn, entity đích, mô tả) có trong đoạn. Kết quả lưu theo chunk_id nên KG cho
     k = 10 / 20 / 30 bài (ý 3) dùng lại, không trích lại.
  2. Gộp (entity resolution): chuẩn hoá tên (chữ thường, bỏ số nhiều, ...), gộp "Tên đầy đủ (VIẾT TẮT)"
     với "VIẾT TẮT" đứng riêng. Mỗi entity/relation giữ danh sách chunk_id nguồn -> truy ngược về văn bản.
  3. Embed: mỗi entity ("tên (loại): mô tả") và relation ("A - B: mô tả") được embed bằng nomic-embed-text,
     dùng để khớp câu hỏi với KG lúc truy vấn (rag_graphrag.py).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError
from tqdm import tqdm

import config
from common import env_info, get_embeddings, get_vectorstore, sha256

# ---------------- 1. Dữ liệu: 30 bài chủ đề quantum ----------------
# Một chủ đề chính (quantum) gồm 2 nhánh gần nhau, để các bài có thực thể / khái niệm chung tạo quan hệ.
TOPICS = ["Quantum Security", "Quantum Machine Learning"]


def load_chunks() -> list[dict]:
    """Chunk của ý 1 (data/chunks.jsonl) + chunk của 7 bài bổ sung (data/kg/chunks_y2.jsonl)."""
    rows = []
    for fp in (config.CHUNKS_FILE, config.KG_CHUNKS_Y2):
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                rows += [json.loads(line) for line in f]
    return rows


def ingest_y2():
    """Chia chunk các PDF trong papers_y2/ đúng như ingest.py làm với papers/ (không đụng dữ liệu ý 1)."""
    import ingest
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    papers_dir, config.PAPERS_DIR = config.PAPERS_DIR, config.PAPERS_Y2_DIR  # load_pdf lấy chủ đề = thư mục con
    try:
        rows = []
        for path in sorted(config.PAPERS_Y2_DIR.rglob("*.pdf")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for j, c in enumerate(splitter.split_documents(ingest.load_pdf(path))):
                rows.append({"text": c.page_content, **c.metadata, "path": f"papers_y2/{c.metadata['path']}",
                             "chunk_id": f"{digest[:12]}-{j}"})
    finally:
        config.PAPERS_DIR = papers_dir
    config.KG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.KG_CHUNKS_Y2, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len({r['source'] for r in rows})} bài -> {len(rows)} chunk -> {config.KG_CHUNKS_Y2}")


def select_papers():
    chunks = load_chunks()
    topic, n = {}, Counter()
    for c in chunks:
        topic[c["source"]] = c["topic"]
        n[c["source"]] += 1
    chosen = [s for s in sorted(topic) if topic[s] in TOPICS]
    assert len(chosen) == 30, len(chosen)
    # Thứ tự phân tầng: xáo trong từng chủ đề rồi xen kẽ theo tỉ lệ -> k bài đầu (k = 10, 20, 30)
    # giữ tỉ lệ chủ đề như 30 bài (ý 3 dựng KG trên k bài đầu).
    rng = random.Random(config.SEED)
    groups = defaultdict(list)
    for s in chosen:
        groups[topic[s]].append(s)
    keyed = []
    for g in groups.values():
        rng.shuffle(g)
        keyed += [((i + 0.5) / len(g), s) for i, s in enumerate(g)]
    order = [s for _, s in sorted(keyed)]
    y2 = {c["source"] for c in chunks if c.get("path", "").startswith("papers_y2/")}
    papers = [{"rank": i + 1, "source": s, "topic": topic[s], "chunks": n[s], "added_for_y2": s in y2}
              for i, s in enumerate(order)]
    config.KG_DIR.mkdir(parents=True, exist_ok=True)
    config.KG_PAPERS_FILE.write_text(json.dumps({
        "rule": "Chủ đề quantum: Quantum Security + Quantum Machine Learning (23 bài trong papers/ + 7 bài bổ sung "
                f"trong papers_y2/); thứ tự phân tầng theo chủ đề (seed {config.SEED}) để k bài đầu giữ tỉ lệ chủ đề",
        "papers": papers}, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in (10, 20, 30):
        print(k, Counter(p["topic"] for p in papers[:k]), sum(p["chunks"] for p in papers[:k]), "chunk")


def load_papers(k: int | None = None) -> list[str]:
    papers = json.loads(config.KG_PAPERS_FILE.read_text(encoding="utf-8"))["papers"]
    return [p["source"] for p in papers[: k or config.KG_K]]


def use_kg_store():
    """Chuyển common.get_vectorstore() sang Chroma riêng của ý 2 (collection kg30, đúng 30 bài).
    Gọi trước khi dựng Vector RAG (rag_langchain) cho ý 2; Chroma của ý 1 không bị đụng tới."""
    config.CHROMA_DIR, config.COLLECTION = config.KG_CHROMA_DIR, config.KG_COLLECTION
    config.CHROMA_SNAPSHOT, config.MANIFEST_FILE = config.KG_CHROMA_SNAPSHOT, config.KG_MANIFEST
    get_vectorstore.cache_clear()


def build_store():
    """Chroma của ý 2: chép nguyên vector của các bài đã có trong Chroma ý 1, embed chunk của bài bổ sung."""
    papers = load_papers(30)
    src = get_vectorstore()._collection  # Chroma ý 1 (chỉ đọc)
    old = src.get(where={"source": {"$in": papers}}, include=["embeddings", "documents", "metadatas"])
    have = {m["source"] for m in old["metadatas"]}
    new = [c for c in load_chunks() if c["source"] in set(papers) - have]
    use_kg_store()
    if config.KG_CHROMA_DIR.exists():
        shutil.rmtree(config.KG_CHROMA_DIR)
    vs = get_vectorstore(auto_restore=False)
    for i in range(0, len(old["ids"]), 512):
        vs._collection.add(ids=old["ids"][i:i + 512], embeddings=old["embeddings"][i:i + 512],
                           documents=old["documents"][i:i + 512], metadatas=old["metadatas"][i:i + 512])
    docs = [Document(page_content=c["text"], metadata={k: v for k, v in c.items() if k != "text"}) for c in new]
    for i in tqdm(range(0, len(docs), 128), desc="Embed chunk bài bổ sung"):
        vs.add_documents(docs[i:i + 128], ids=[d.metadata["chunk_id"] for d in docs[i:i + 128]])
    n = vs._collection.count()
    get_vectorstore.cache_clear()
    with zipfile.ZipFile(config.KG_CHROMA_SNAPSHOT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(config.KG_CHROMA_DIR.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(config.DATA_DIR))
    config.KG_MANIFEST.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"), "papers": len(papers),
        "papers_from_y1_chroma": len(have), "papers_added": len(set(papers) - have),
        "chunks": n, "collection": config.KG_COLLECTION, "distance": "cosine",
        "chunking": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                     "drop_references": config.DROP_REFERENCES},
        "files": {"chroma.zip": sha256(config.KG_CHROMA_SNAPSHOT), "chunks_y2.jsonl": sha256(config.KG_CHUNKS_Y2)},
        "environment": env_info()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chroma ý 2: {n} chunk ({len(have)} bài chép từ ý 1 + {len(set(papers) - have)} bài mới) "
          f"-> {config.KG_CHROMA_SNAPSHOT} ({config.KG_CHROMA_SNAPSHOT.stat().st_size / 1e6:.1f} MB)")


# ---------------- 2. Trích xuất entity / relation ----------------
TYPES = {"ALGORITHM", "PROTOCOL", "ATTACK", "DEFENSE", "SYSTEM", "CONCEPT", "METRIC", "DATASET",
         "ORGANIZATION", "STANDARD", "HARDWARE"}

# Định dạng dòng "E|...", "R|..." thay JSON: ít token đầu ra hơn ~40% (trích xuất là bước tốn thời gian nhất).
# Ví dụ minh hoạ lấy ngoài các chủ đề của 30 bài.
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You build a knowledge graph from a passage of a scientific paper.\n"
     "Output one line per item and nothing else:\n"
     "E|<entity name>|<TYPE>|<one short sentence describing the entity, from the passage>\n"
     "R|<entity name>|<entity name>|<one sentence from the passage that names BOTH entities and states how "
     "they are related>\n"
     "Rules:\n"
     "- Entities are the important technical things: algorithms, protocols, attacks, defenses, systems/tools, "
     "concepts, metrics, datasets, organizations, standards, hardware. Skip authors, citation numbers, figures, "
     "tables, people and generic words ('approach', 'method', 'results', 'this paper').\n"
     f"- TYPE is one of: {', '.join(sorted(TYPES))}.\n"
     "- Use the full common name. If the passage gives an acronym, write 'Full Name (ACRONYM)'.\n"
     "- Relations only between entities you listed (same names) and only if the passage states them.\n"
     "- At most 12 entities and 12 relations. If the passage has no technical content, output NONE.\n\n"
     "Example passage: \"The Border Gateway Protocol (BGP) exchanges routes between autonomous systems. "
     "BGP hijacking announces another network's prefixes; Resource Public Key Infrastructure (RPKI) "
     "lets routers reject such invalid announcements.\"\n"
     "Example output:\n"
     "E|Border Gateway Protocol (BGP)|PROTOCOL|Protocol that exchanges routes between autonomous systems.\n"
     "E|autonomous system|CONCEPT|Network that exchanges routes with others via BGP.\n"
     "E|BGP hijacking|ATTACK|Attack that announces another network's prefixes.\n"
     "E|Resource Public Key Infrastructure (RPKI)|DEFENSE|Lets routers reject invalid route announcements.\n"
     "R|Border Gateway Protocol (BGP)|autonomous system|BGP exchanges routes between autonomous systems.\n"
     "R|BGP hijacking|Border Gateway Protocol (BGP)|BGP hijacking abuses BGP announcements.\n"
     "R|Resource Public Key Infrastructure (RPKI)|BGP hijacking|RPKI lets routers reject hijacked announcements."),
    ("human", "Paper: {title}\n\nPassage:\n{text}"),
])


def has_content(text: str) -> bool:
    """Bỏ qua chunk rác (bảng số, công thức vỡ, quá ngắn): không gọi LLM."""
    return len(text) >= 150 and sum(ch.isalpha() for ch in text) / len(text) > 0.5


def parse_lines(raw: str) -> tuple[list, list]:
    ents, rels = [], []
    for line in raw.splitlines():
        parts = [p.strip().strip("\"'`*") for p in line.strip().strip("-• ").split("|")]
        if len(parts) < 4 or not parts[1] or not parts[2]:
            continue
        tag, a, b, desc = parts[0].upper(), parts[1], parts[2], "|".join(parts[3:]).strip()
        if tag == "E":
            t = b.upper().replace(" ", "_")
            ents.append([a, t if t in TYPES else "CONCEPT", desc])
        elif tag == "R" and a.lower() != b.lower():
            rels.append([a, b, desc])
    # chỉ giữ relation giữa 2 entity đã liệt kê (model 7B hay nối tới tác giả / thứ không phải entity)
    names = {norm(x) for n, _, _ in ents for x in (n, *split_acronym(n)) if x}
    rels = [r for r in rels if all(norm(split_acronym(x)[0]) in names or norm(x) in names for x in r[:2])]
    return ents[:20], rels[:20]


def load_extractions() -> dict:
    if not config.KG_EXTRACTIONS.exists():
        return {}
    with open(config.KG_EXTRACTIONS, encoding="utf-8") as f:
        return {d["chunk_id"]: d for d in map(json.loads, f)}


def extract(k: int):
    papers = load_papers(k)
    rank = {s: i for i, s in enumerate(papers)}
    todo_chunks = sorted((c for c in load_chunks() if c["source"] in rank),
                         key=lambda c: (rank[c["source"]], c["page"]))   # theo thứ tự bài -> k nhỏ xong trước
    done = load_extractions()
    todo = [c for c in todo_chunks if c["chunk_id"] not in done]
    print(f"{k} bài, {len(todo_chunks)} chunk, đã có {len(todo_chunks) - len(todo)}, còn {len(todo)}")
    # num_predict giới hạn đầu ra (12 entity + 12 relation ~ 600 token) để tránh lặp vô hạn
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, temperature=0, seed=config.SEED,
                     num_ctx=config.NUM_CTX, num_predict=800,
                     num_gpu=config.KG_NUM_GPU if config.KG_NUM_GPU >= 0 else None)
    chain = EXTRACT_PROMPT | llm
    with open(config.KG_EXTRACTIONS, "a", encoding="utf-8") as f:
        for c in tqdm(todo, desc="Trích entity/relation"):
            rec = {"chunk_id": c["chunk_id"], "source": c["source"], "model": config.LLM_MODEL}
            if not has_content(c["text"]):
                rec.update(entities=[], relations=[], skipped=True)
            else:
                try:
                    out = chain.invoke({"title": c["source"], "text": c["text"]})
                    ents, rels = parse_lines(out.content)
                    rec.update(entities=ents, relations=rels, out_tokens=out.response_metadata.get("eval_count"))
                except ResponseError as e:
                    rec.update(entities=[], relations=[], error=str(e)[:200])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ---------------- 3. Gộp thành KG ----------------
ACRONYM = re.compile(r"^(.*?)\s*\(([^()]+)\)$")
GENERIC_SUFFIX = {"algorithm", "algorithms", "scheme", "schemes", "protocol", "protocols", "framework",
                  "technique", "techniques", "method", "methods", "approach", "mechanism", "mechanisms"}


def norm(name: str) -> str:
    s = name.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-").lower()
    s = re.sub(r"[\"`*]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;'-")
    words = s.split(" ")
    if len(words) > 1 and words[-1] in GENERIC_SUFFIX:   # "CRYSTALS-Kyber algorithm" -> "crystals-kyber"
        words = words[:-1]
    w = words[-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):   # honeypots -> honeypot
        words[-1] = w[:-3] + "y" if w.endswith("ies") else w[:-1]
    return " ".join(words)


def split_acronym(name: str) -> tuple[str, str | None]:
    """'Quantum Key Distribution (QKD)' -> ('Quantum Key Distribution', 'QKD'); đảo lại nếu viết ngược."""
    m = ACRONYM.match(name.strip())
    if not m or not m.group(1):
        return name.strip(), None
    full, short = m.group(1).strip(), m.group(2).strip()
    if len(short) > len(full):
        full, short = short, full
    return (full, short) if len(short) <= 12 else (name.strip(), None)


def build(k: int):
    papers = load_papers(k)
    allowed = set(papers)
    chunk_source = {c["chunk_id"]: c["source"] for c in load_chunks() if c["source"] in allowed}
    ex = load_extractions()
    missing = [cid for cid in chunk_source if cid not in ex]
    if missing:
        raise SystemExit(f"Còn {len(missing)} chunk chưa trích xuất: chạy python kg_build.py --extract --k {k}")

    # Bảng viết tắt -> tên đầy đủ (chỉ dùng khi viết tắt đó ứng với đúng 1 tên đầy đủ)
    acr = defaultdict(set)
    for cid in chunk_source:
        for name, _, _ in ex[cid]["entities"]:
            full, short = split_acronym(name)
            if short:
                acr[norm(short)].add(norm(full))
    acr = {s: next(iter(f)) for s, f in acr.items() if len(f) == 1}

    def key_of(name: str) -> str:
        full, short = split_acronym(name)
        key = norm(full)
        return acr.get(key, key)

    ents = {}  # key -> dict

    def entity(name: str, etype: str | None = None):
        key = key_of(name)
        e = ents.setdefault(key, {"names": Counter(), "types": Counter(), "descriptions": [], "aliases": set(),
                                  "chunk_ids": set(), "sources": set()})
        full, short = split_acronym(name)
        e["names"][full] += 1
        if short:
            e["aliases"].add(short)
        if etype:
            e["types"][etype] += 1
        return key, e

    rels = {}  # frozenset(key_a, key_b) -> dict
    for cid, src in chunk_source.items():
        for name, etype, desc in ex[cid]["entities"]:
            _, e = entity(name, etype)
            if desc and desc not in e["descriptions"]:
                e["descriptions"].append(desc)
            e["chunk_ids"].add(cid)
            e["sources"].add(src)
        for a, b, desc in ex[cid]["relations"]:
            (ka, ea), (kb, eb) = entity(a), entity(b)   # entity chỉ xuất hiện trong relation vẫn được thêm
            if ka == kb:
                continue
            for e in (ea, eb):
                e["chunk_ids"].add(cid)
                e["sources"].add(src)
            r = rels.setdefault(frozenset((ka, kb)), {"pair": (ka, kb), "descriptions": [], "chunk_ids": set(),
                                                      "sources": set()})
            if desc and desc not in r["descriptions"]:
                r["descriptions"].append(desc)
            r["chunk_ids"].add(cid)
            r["sources"].add(src)

    keys = sorted(ents)
    idx = {key: i for i, key in enumerate(keys)}
    entities = [{"id": idx[key], "key": key, "name": e["names"].most_common(1)[0][0],
                 "type": e["types"].most_common(1)[0][0] if e["types"] else "CONCEPT",
                 "aliases": sorted(e["aliases"]), "descriptions": e["descriptions"],
                 "chunk_ids": sorted(e["chunk_ids"]), "sources": sorted(e["sources"])}
                for key, e in ((key, ents[key]) for key in keys)]
    relations = [{"id": i, "source": idx[r["pair"][0]], "target": idx[r["pair"][1]],
                  "descriptions": r["descriptions"], "chunk_ids": sorted(r["chunk_ids"]),
                  "sources": sorted(r["sources"])}
                 for i, r in enumerate(sorted(rels.values(), key=lambda r: r["pair"]))]

    # Embed entity + relation để khớp câu hỏi lúc truy vấn
    emb = get_embeddings()
    e_txt = [entity_text(e) for e in entities]
    r_txt = [relation_text(r, entities) for r in relations]
    vec = lambda txts: np.asarray(  # noqa: E731
        [v for i in tqdm(range(0, len(txts), 64), desc="Embed")
         for v in emb.embed_documents(txts[i:i + 64])], dtype=np.float16)
    out = config.KG_DIR / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    stats = graph_stats(entities, relations, chunk_source, ex)
    (out / "graph.json").write_text(json.dumps({"k": k, "papers": papers, "stats": stats, "entities": entities,
                                                "relations": relations}, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "vectors.npz", entities=vec(e_txt), relations=vec(r_txt))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def entity_text(e: dict, max_chars: int = 300) -> str:
    return f"{e['name']} ({e['type']}): {'; '.join(e['descriptions'])}"[:max_chars]


def relation_text(r: dict, entities: list[dict], max_chars: int = 300) -> str:
    return (f"{entities[r['source']]['name']} - {entities[r['target']]['name']}: "
            f"{'; '.join(r['descriptions'])}")[:max_chars]


def graph_stats(entities, relations, chunk_source, ex) -> dict:
    parent = list(range(len(entities)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in relations:
        parent[find(r["source"])] = find(r["target"])
    comp = Counter(find(i) for i in range(len(entities)))
    deg = Counter()
    for r in relations:
        deg[r["source"]] += 1
        deg[r["target"]] += 1
    n = max(len(entities), 1)
    return {"papers": len(set(chunk_source.values())), "chunks": len(chunk_source),
            "chunks_skipped": sum(bool(ex[c].get("skipped")) for c in chunk_source),
            "chunks_error": sum(bool(ex[c].get("error")) for c in chunk_source),
            "raw_entity_mentions": sum(len(ex[c]["entities"]) for c in chunk_source),
            "raw_relation_mentions": sum(len(ex[c]["relations"]) for c in chunk_source),
            "entities": len(entities), "relations": len(relations),
            "entities_in_2plus_papers": sum(len(e["sources"]) >= 2 for e in entities),
            "relations_in_2plus_papers": sum(len(r["sources"]) >= 2 for r in relations),
            "isolated_entities": sum(deg[i] == 0 for i in range(len(entities))),
            "avg_degree": round(2 * len(relations) / n, 3),
            "components": len(comp), "largest_component": max(comp.values(), default=0),
            "entity_types": dict(Counter(e["type"] for e in entities).most_common())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true", help="chia chunk papers_y2/ -> data/kg/chunks_y2.jsonl")
    ap.add_argument("--select", action="store_true", help="chọn 30 bài -> data/kg/papers.json")
    ap.add_argument("--store", action="store_true", help="Chroma riêng của ý 2 -> data/kg/chroma.zip")
    ap.add_argument("--extract", action="store_true", help="trích entity/relation (cache theo chunk)")
    ap.add_argument("--build", action="store_true", help="gộp thành KG + embed -> data/kg/k<k>/")
    ap.add_argument("--k", type=int, default=config.KG_K, help="dùng k bài đầu trong papers.json")
    args = ap.parse_args()
    if args.ingest:
        ingest_y2()
    if args.select:
        select_papers()
    if args.store:
        build_store()
    if args.extract:
        extract(args.k)
    if args.build:
        build(args.k)
    if not (args.ingest or args.select or args.store or args.extract or args.build):
        ap.print_help()


if __name__ == "__main__":
    main()
