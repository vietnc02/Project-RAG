"""Ý 2 — GraphRAG: truy xuất qua Knowledge Graph (local search) -> generate, 1 lần gọi LLM.

    python rag_graphrag.py "Which signature schemes did NIST select for standardization?"

KG dựng sẵn bởi kg_build.py (data/kg/k<KG_K>/). Truy xuất (theo hướng local search của Microsoft GraphRAG /
LightRAG, không cần gọi LLM):
  1. Khớp entity: cosine(câu hỏi, vector entity) + điểm cộng nếu tên / viết tắt của entity xuất hiện
     nguyên văn trong câu hỏi -> KG_SEED_ENTITIES entity hạt giống.
  2. Mở rộng 1 bước: các quan hệ nối với entity hạt giống (+ vài quan hệ gần câu hỏi nhất toàn KG),
     xếp theo cosine(câu hỏi, vector quan hệ) -> KG_MAX_RELATIONS quan hệ.
  3. Chọn văn bản: mỗi chunk được cộng điểm của các entity hạt giống / quan hệ đã chọn có nguồn ở chunk đó
     -> TOP_K chunk điểm cao nhất (cùng số chunk với Vector RAG).
Context đưa vào LLM = mô tả entity + quan hệ (từ KG) + TOP_K đoạn văn. Prompt sinh câu trả lời và LLM
giống hệt Vector RAG (common.GENERATE_PROMPT) -> khác biệt chỉ đến từ cách truy xuất.
"""
import json
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
from langchain_core.output_parsers import StrOutputParser

import config
from common import GENERATE_PROMPT, get_embeddings, get_llm

NAME_BONUS = 0.15        # điểm cộng khi tên entity xuất hiện nguyên văn trong câu hỏi
GLOBAL_RELATIONS = 5     # số quan hệ lấy thêm theo cosine trên toàn KG (không cần nối với entity hạt giống)


class KnowledgeGraph:
    def __init__(self, k: int):
        d = config.KG_DIR / f"k{k}"
        if not (d / "graph.json").exists():
            raise SystemExit(f"Chưa có KG {d}: chạy python kg_build.py --extract --build --k {k}")
        g = json.loads((d / "graph.json").read_text(encoding="utf-8"))
        v = np.load(d / "vectors.npz")
        self.k, self.entities, self.relations = k, g["entities"], g["relations"]
        self.ev, self.rv = (self._unit(v["entities"]), self._unit(v["relations"]))
        self.adj = defaultdict(list)
        for r in self.relations:
            self.adj[r["source"]].append(r["id"])
            self.adj[r["target"]].append(r["id"])
        from kg_build import load_chunks  # chunk ý 1 + chunk của 7 bài bổ sung
        chunks = {c["chunk_id"]: c for c in load_chunks()}
        needed = {cid for x in (*self.entities, *self.relations) for cid in x["chunk_ids"]}
        self.chunks = {cid: chunks[cid] for cid in needed}
        # tên / viết tắt đủ đặc trưng để khớp nguyên văn (bỏ tên quá ngắn, quá chung)
        self.name_patterns = []
        for e in self.entities:
            for name in {e["name"], *e["aliases"]}:
                if len(name) >= 4 or (len(name) >= 2 and name.isupper()):
                    self.name_patterns.append((e["id"], re.compile(rf"(?<!\w){re.escape(name)}s?(?!\w)",
                                                                   0 if name.isupper() else re.I)))

    @staticmethod
    def _unit(m):
        m = np.asarray(m, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    def search(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or config.TOP_K
        q = np.asarray(get_embeddings().embed_query(question), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9

        # 1. entity hạt giống
        e_score = self.ev @ q
        sims = e_score.copy()
        for eid, pat in self.name_patterns:
            if pat.search(question):
                e_score[eid] = sims[eid] + NAME_BONUS
        seeds = [int(i) for i in np.argsort(-e_score)[: config.KG_SEED_ENTITIES]]

        # 2. quan hệ: nối với entity hạt giống + vài quan hệ gần câu hỏi nhất toàn KG
        r_sim = self.rv @ q
        cand = {rid for e in seeds for rid in self.adj[e]}
        cand |= {int(i) for i in np.argsort(-r_sim)[:GLOBAL_RELATIONS]}
        rels = sorted(cand, key=lambda rid: -r_sim[rid])[: config.KG_MAX_RELATIONS]

        # 3. chunk: cộng điểm entity hạt giống + quan hệ đã chọn có nguồn tại chunk
        c_score = defaultdict(float)
        for e in seeds:
            for cid in self.entities[e]["chunk_ids"]:
                c_score[cid] += float(e_score[e])
        for rid in rels:
            for cid in self.relations[rid]["chunk_ids"]:
                c_score[cid] += float(r_sim[rid])
        chunk_ids = sorted(c_score, key=lambda c: -c_score[c])[:top_k]
        return {"entities": [self.entities[e] for e in seeds], "relations": [self.relations[r] for r in rels],
                "chunks": [self.chunks[c] for c in chunk_ids]}

    def kg_text(self, hit: dict, max_chars: int = 220) -> str:
        ents = [f"- {e['name']} ({e['type']}): {'; '.join(e['descriptions'])[:max_chars]}" for e in hit["entities"]]
        rels = [f"- {self.entities[r['source']]['name']} -- {self.entities[r['target']]['name']}: "
                f"{'; '.join(r['descriptions'])[:max_chars]}" for r in hit["relations"]]
        return "Knowledge graph entities:\n" + "\n".join(ents) + "\n\nKnowledge graph relations:\n" + "\n".join(rels)


@lru_cache
def get_kg(k: int | None = None) -> KnowledgeGraph:
    return KnowledgeGraph(k or config.KG_K)


def format_context(kg_text: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[{i + 1}] (source: {c['source']}, page {c['page']})\n{c['text']}"
                           for i, c in enumerate(chunks))
    return f"{kg_text}\n\nText passages:\n{passages}"


def answer(question: str) -> dict:
    kg = get_kg()
    t0 = time.perf_counter()
    hit = kg.search(question)
    kg_text = kg.kg_text(hit)
    out = (GENERATE_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"context": format_context(kg_text, hit["chunks"]), "question": question})
    return {
        "answer": out.strip(),
        "contexts": [{k: c[k] for k in ("text", "source", "topic", "page", "chunk_id")} for c in hit["chunks"]],
        "kg_context": kg_text,
        "kg_entities": [e["name"] for e in hit["entities"]],
        "kg_relations": [[kg.entities[r["source"]]["name"], kg.entities[r["target"]]["name"]]
                         for r in hit["relations"]],
        "latency_s": round(time.perf_counter() - t0, 2),
        "llm_calls": 1,
        "trace": ["match_entities", "expand_relations", "select_chunks", "generate"],
    }


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]) or "Which signature schemes did NIST select for standardization?")
    print(r["answer"])
    print("\nEntity:", r["kg_entities"])
    print("Nguồn:", sorted({c["source"] for c in r["contexts"]}))
