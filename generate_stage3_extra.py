"""Supplement stage 3 with entity-bridge questions and near-topic negative candidates.

All generated rows remain unreviewed. Separate review sheets preserve the first S3 review.
The cosine cap is a declared sampling constraint, not a measurement of answer quality.
"""
import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ollama import ResponseError

import config
from common import env_info, get_embeddings, get_llm, get_vectorstore
from evaluate import verify
from evaluate_graphrag import extract_claims
from generate_testset import ALONE_PROMPT, good_chunk
from kg_build import load_chunks, load_papers, use_kg_store
from stage3_review import apply_review_files, read_json

DIR = config.PROJECT_DIR / "testset"
RAW = {"bridge": DIR / "multihop_s3_bridge_raw.json", "near": DIR / "unanswerable_s3_near_raw.json"}
BAD_QUESTION = re.compile(r"\bpassage\b|\bpaper [ab12]\b|\bthis (paper|study|work)\b|\bthe authors\b", re.I)
BRIDGE = ChatPromptTemplate.from_messages([
    ("system", "Create one self-contained scientific multi-hop question. The passages share the named entity. "
     "Use it as a meaningful bridge; answering must require a distinct fact from EACH passage. "
     "Do not merely ask two unrelated questions, presume an unsupported causal link, or mention passage/paper A/B. "
     "If no defensible connection exists return {{\"skip\":true}}. Otherwise return JSON with question, "
     "ground_truth (2-4 sentences supported only by both passages), evidence_a and evidence_b "
     "(exact short quotations supporting the respective facts)."),
    ("human", "Shared entity: {entity}\nTitle A: {title_a}\n{text_a}\nTitle B: {title_b}\n{text_b}")])
NEAR = ChatPromptTemplate.from_messages([
    ("system", "Propose a plausible, self-contained question close to the technical topic in the passage, "
     "but asking for a specific result or condition NOT provided there. Do not invent a false premise, "
     "make an absurd question, or mention a passage. It will later be checked against the full corpus by "
     "retrieval and a human; you cannot declare it unanswerable for the entire corpus. "
     "Return JSON with question and missing_information (what additional evidence would be needed), "
     "or {{\"skip\":true}}."),
    ("human", "Title: {title}\n{text}")])


def load_existing(kind):
    return read_json(RAW[kind]) if RAW[kind].exists() else []


def save(kind, rows):
    tmp = RAW[kind].with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(RAW[kind])


def text_norm(text):
    return " ".join(text.casefold().split())


def quote_in(quote, text):
    return isinstance(quote, str) and len(quote.strip()) >= 15 and text_norm(quote) in text_norm(text)


def candidates(graph, chunks, seed):
    """Pair by shared entity provenance, without nearest-neighbour selection."""
    rng = random.Random(seed)
    entities = list(graph["entities"])
    rng.shuffle(entities)
    seen = set()
    for entity in entities:
        if entity.get("type") not in ("ALGORITHM", "PROTOCOL", "CONCEPT", "METHOD", "SYSTEM"):
            continue
        names = [entity["name"], *entity.get("aliases", [])]
        pool = [chunks[c] for c in entity["chunk_ids"] if c in chunks and any(
            len(n) >= 3 and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", chunks[c]["text"], re.I)
            for n in names)]
        rng.shuffle(pool)
        # Bound pairs for broad entities, so generic concepts cannot dominate the candidate pool.
        count = 0
        for a, b in itertools.combinations(pool, 2):
            pair = tuple(sorted((a["chunk_id"], b["chunk_id"])))
            if a["source"] == b["source"] or pair in seen:
                continue
            seen.add(pair)
            yield entity, a, b
            count += 1
            if count >= 12:
                break


def strict_complete(chain, text, question):
    try:
        value = json.loads(chain.invoke({"text": text, "question": question})).get("complete")
    except (json.JSONDecodeError, ResponseError):  # JSON hỏng / Ollama huỷ do lặp token
        return None
    return value.strip().lower() if isinstance(value, str) and value.strip().lower() in ("yes", "no") else None


def generate(kind, target, max_attempts, seed, cosine_cap):
    rows = load_existing(kind)
    if len(rows) >= target:
        print(f"{kind}: already {len(rows)}/{target}", flush=True)
        return
    papers = set(load_papers(30))
    chunks = {c["chunk_id"]: c for c in load_chunks() if c["source"] in papers and good_chunk(c)}
    graph = read_json(config.KG_DIR / "k30/graph.json")
    gen = (BRIDGE if kind == "bridge" else NEAR) | get_llm(json_mode=True, seed=seed) | StrOutputParser()
    alone = ALONE_PROMPT | get_llm(json_mode=True, seed=seed) | StrOutputParser()
    emb, vectors = get_embeddings(), {}
    existing_questions = {text_norm(t["question"]) for fp in DIR.glob("*.json")
                          if fp.name in ("testset.json", "multihop.json", "multihop_kg.json", "unanswerable.json")
                          or "s3" in fp.name
                          for t in read_json(fp) if isinstance(t, dict) and "question" in t}
    used = {tuple(sorted(t.get("chunk_ids", []))) for t in rows}
    entity_counts = Counter(t.get("bridge_entity") for t in rows)
    source_pairs = Counter(tuple(sorted(t.get("sources", []))) for t in rows)
    history = RAW[kind].with_suffix(".attempts.jsonl")
    tried = {r["candidate"] for r in (json.loads(s) for s in history.read_text(encoding="utf-8").splitlines())} if history.exists() else set()
    if kind == "bridge":
        use_kg_store()
        cached = get_vectorstore().get(include=["embeddings", "metadatas"])
        for metadata, value in zip(cached["metadatas"], cached["embeddings"]):
            v = np.asarray(value, float)
            vectors[metadata["chunk_id"]] = v / np.linalg.norm(v)
        pool = list(candidates(graph, chunks, seed))
        random.Random(seed).shuffle(pool)
    else:
        y1_store = get_vectorstore()
        use_kg_store()
        y2_store = get_vectorstore()
        pool = list(chunks.values())
        random.Random(seed).shuffle(pool)
    attempts, rejected = 0, Counter()
    for candidate in pool:
        if len(rows) >= target or attempts >= max_attempts:
            break
        if kind == "bridge":
            entity, a, b = candidate
            pair = tuple(sorted((a["chunk_id"], b["chunk_id"])))
            candidate_id = f"{entity['id']}:{':'.join(pair)}"
            if pair in used or entity_counts[entity["name"]] >= 5 or source_pairs[tuple(sorted((a["source"], b["source"])))] >= 2:
                continue
        else:
            a = candidate
            candidate_id = a["chunk_id"]
        if candidate_id in tried:
            continue
        attempts += 1
        reason, checks = "invalid", []
        if kind == "bridge":
            for c in (a, b):
                if c["chunk_id"] not in vectors:
                    v = np.asarray(emb.embed_documents([c["text"]])[0], float)
                    vectors[c["chunk_id"]] = v / np.linalg.norm(v)
            cosine = float(vectors[a["chunk_id"]] @ vectors[b["chunk_id"]])
            if cosine > cosine_cap:
                reason = "cosine_above_cap"
                data = {}
            else:
                try:
                    data = json.loads(gen.invoke({"entity": entity["name"], "title_a": a["source"], "text_a": a["text"],
                                                  "title_b": b["source"], "text_b": b["text"]}))
                except (json.JSONDecodeError, ResponseError):
                    data = {}
        else:
            try:
                data = json.loads(gen.invoke({"title": a["source"], "text": a["text"]}))
            except (json.JSONDecodeError, ResponseError):
                data = {}
        q = data.get("question", "")
        if isinstance(q, str) and q.strip() and not data.get("skip") and not BAD_QUESTION.search(q) and text_norm(q) not in existing_questions:
            if kind == "bridge":
                gt = data.get("ground_truth")
                checks = []
                if isinstance(gt, str) and gt.strip() and quote_in(data.get("evidence_a"), a["text"]) and quote_in(data.get("evidence_b"), b["text"]):
                    checks = [strict_complete(alone, c["text"], q) for c in (a, b)]
                    if checks == ["no", "no"]:
                        checks.append(strict_complete(alone, a["text"] + "\n\n" + b["text"], q))
                    if checks == ["no", "no", "yes"]:
                        # mọi ý của đáp án phải được hai đoạn hỗ trợ (judge khác họ với mô hình sinh); loại đáp án
                        # gán nhầm thông tin giữa hai đoạn hoặc thêm chi tiết không có (như ứng viên thử b001)
                        claims = extract_claims(q, gt)
                        supported, judged = verify(claims, [a, b])
                        checks.append(f"gt_supported {supported}/{len(claims)}")
                        if not (judged == len(claims) and supported == len(claims)):
                            checks = []
                if len(checks) == 4:
                    row = {"id": f"b{len(rows)+1:03d}", "type": "multihop", "question": q.strip(), "ground_truth": gt.strip(),
                           "sources": [a["source"], b["source"]], "chunk_ids": [a["chunk_id"], b["chunk_id"]],
                           "chunk_texts": [a["text"], b["text"]], "bridge_entity": entity["name"],
                           "sampling": "shared_entity_cosine_cap", "pair_cosine": cosine, "cosine_cap": cosine_cap,
                           "evidence_quotes": [data["evidence_a"], data["evidence_b"]], "alone_checks": checks}
                    used.add(pair)
                    entity_counts[entity["name"]] += 1
                    source_pairs[tuple(sorted(row["sources"]))] += 1
                    reason = "accepted"
                else:
                    reason = "evidence_or_multihop_check_failed"
            else:
                docs = {d.metadata["chunk_id"]: d for store in (y1_store, y2_store) for d in store.similarity_search(q, k=8)}
                checks = [strict_complete(alone, d.page_content, q) for d in docs.values()]
                # This is only a screen. A human must still establish corpus-level absence.
                if checks and all(c == "no" for c in checks) and data.get("missing_information"):
                    row = {"id": f"u3{len(rows)+1:03d}", "type": "unanswerable", "question": q.strip(),
                           "ground_truth": "I cannot find the answer in the provided documents.",
                           "missing_information": data["missing_information"], "anchor_source": a["source"],
                           "anchor_text": a["text"], "screened_contexts": [{"text": d.page_content, **d.metadata} for d in docs.values()],
                           "absence_verified": False, "sampling": "near_topic_candidate"}
                    reason = "accepted"
                else:
                    reason = "retrieved_evidence_or_unknown_verdict"
            if reason == "accepted":
                row.update(reviewed=False, keep=True, generation_seed=seed, topic=a["topic"])
                rows.append(row)
                existing_questions.add(text_norm(q))
                save(kind, rows)
                print(f"{kind}: {len(rows)}/{target}, attempt {attempts}", flush=True)
        rejected[reason] += 1
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"candidate": candidate_id, "result": reason, "seed": seed,
                                "cosine_cap": cosine_cap, "checks": checks, "generated": data}, ensure_ascii=False) + "\n")
    info = {"target": target, "generated": len(rows), "attempts_this_run": attempts, "outcomes": dict(rejected),
            "seed": seed, "cosine_cap": cosine_cap, "environment": env_info(), "reviewed": False}
    RAW[kind].with_suffix(".run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)


def review_dir(kind):
    return DIR / f"review_s3_{kind}"


def sheet(kind):
    rows = load_existing(kind)
    if not rows:
        return
    rd = review_dir(kind)
    rd.mkdir(exist_ok=True)
    cases, key = [], {}
    order = random.Random(config.SEED).sample(rows, len(rows))
    for i, t in enumerate(order, 1):
        code = f"Q{i:03d}"
        key[code] = t["id"]
        blocks = [{"title": "Câu hỏi", "text": t["question"]}, {"title": "Đáp án", "text": t["ground_truth"]}]
        if kind == "bridge":
            blocks += [{"title": "Entity chung", "text": t["bridge_entity"]}]
            blocks += [{"title": s, "text": text} for s, text in zip(t["sources"], t["chunk_texts"])]
        else:
            blocks += [{"title": "Thông tin cần nhưng chưa tìm thấy", "text": t["missing_information"]}]
            blocks += [{"title": d["source"], "text": d["text"]} for d in t["screened_contexts"]]
        blocks += [{"key": "keep", "group": "Duyệt", "prompt": "Câu này:",
                    "options": [[1, "Giữ", "y"], [2, "Giữ, cần sửa", ""], [0, "Loại", "n"]]}]
        cases.append({"code": code, "title": t["type"], "blocks": blocks})
    digest = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()[:12]
    key_file = rd / "key.json"
    if key_file.exists() and read_json(key_file)["sheet_id"] != digest:
        raise SystemExit(f"Phiếu {kind} đã tồn tại; không ghi đè phiếu đang duyệt. Hoàn tất sinh trước khi tạo phiếu.")
    guide = "Multi-hop phải cần cả hai nguồn; kiểm tra đáp án và mối liên hệ qua entity."
    if kind == "near":
        guide = "Chỉ giữ khi đã kiểm tra toàn corpus và xác nhận thiếu thông tin; top-8 không chứng minh không có đáp án. Loại câu có tiền đề sai."
    payload = {"sheet_id": digest, "title": f"Duyệt S3 {kind}", "file": f"human_labels_s3_{kind}.json",
               "guide": [["Kiểm tra nguồn", guide], ["Cần sửa", "Ghi đầy đủ câu hỏi và đáp án thay thế."]], "cases": cases}
    html = (DIR / "review/template.html").read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"), 1)
    (rd / "phieu_duyet.html").write_text(html, encoding="utf-8")
    key_file.write_text(json.dumps({"sheet_id": digest, "cases": key}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=list(RAW), required=True)
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    # 25th percentile ~0.736 in a seeded 200-pair calibration, measured before judging answers.
    ap.add_argument("--cosine-cap", type=float, default=0.74)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--labels", type=Path)
    args = ap.parse_args()
    if args.generate:
        generate(args.kind, args.target, args.max_attempts, args.seed, args.cosine_cap)
    if args.sheet:
        sheet(args.kind)
    if args.labels:
        rd = review_dir(args.kind)
        dst = RAW[args.kind].with_name(RAW[args.kind].name.replace("_raw", ""))
        print(apply_review_files([(RAW[args.kind], dst)], rd / "key.json", args.labels, rd / "edits.json", config.RESULTS_DIR))


if __name__ == "__main__":
    main()
