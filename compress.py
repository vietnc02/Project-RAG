"""Ý 4: nén context của GraphRAG (mô tả KG + đoạn văn) để giảm token mà không mất thông tin quan trọng.

    python compress.py "Which signature schemes did NIST select for standardization?" --mode both
    python compress.py --calibrate          # tỉ lệ giảm ký tự trên 85 câu của ý 2 theo từng mức nén (không gọi LLM)

Truy xuất giữ nguyên GraphRAG của ý 2 (rag_graphrag.search: cùng 8 entity, 12 quan hệ, 4 đoạn văn). Nén chỉ bằng cách
XOÁ BỚT (extractive): context nén luôn là tập con của context gốc, không thêm thông tin mới, không gọi thêm LLM.
  - nén KG  : giữ tỉ lệ KG_KEEP entity / quan hệ gần câu hỏi nhất (cosine embedding); mỗi dòng còn lại chỉ giữ các
              mảnh mô tả gần câu hỏi nhất, bỏ mảnh trùng lặp, trong giới hạn KG_ITEM_CHARS ký tự (luôn giữ ít nhất
              1 mảnh). Mảnh lấy từ đúng phần mô tả mà context gốc hiển thị (220 ký tự đầu), nên không đưa vào mô
              tả nào context gốc không có.
  - nén văn bản: bỏ nhiễu (đánh số trích dẫn [12], URL, email, dòng arXiv), tách câu, giữ tỉ lệ SENT_KEEP số câu
              gần câu hỏi nhất của mỗi đoạn (luôn giữ ít nhất 1 câu), giữ nguyên thứ tự câu.
Chế độ: base (không nén) / kg / text / both. Prompt sinh câu trả lời + LLM giữ nguyên (common.GENERATE_PROMPT).

Baseline: cùng số dòng KG / số câu / giới hạn ký tự như cách chọn theo độ liên quan,
chỉ khác cách chọn, để biết việc chọn theo độ liên quan có giá trị riêng không:
  - rand_<kg|text|both>[_s<seed>]: chọn ngẫu nhiên (seed theo câu hỏi + seed của chế độ)
  - head_<kg|text|both>          : giữ phần đầu (dòng KG đầu tiên, mảnh mô tả đầu tiên, câu đầu tiên)
  - both_l25, both_l40           : nén cả hai theo độ liên quan ở hai mức nhẹ hơn (token reduction ~25% và ~40%)
"""
import argparse
import hashlib
import re
import sys
import time
from functools import lru_cache

import numpy as np
from langchain_core.output_parsers import StrOutputParser

import config
from common import GENERATE_PROMPT, get_embeddings, get_llm
from rag_graphrag import format_context, get_kg

MODES = {"base": (False, False), "kg": (True, False), "text": (False, True), "both": (True, True)}
PARTS = {"kg": (True, False), "text": (False, True), "both": (True, True)}
# mức nén thêm cho chế độ nén cả hai: (KG_KEEP, SENT_KEEP, KG_ITEM_CHARS), chọn bằng --calibrate theo độ dài
LEVELS = {"both_l25": (0.75, 0.75, 165), "both_l40": (0.6, 0.6, 130)}
BASELINE_MODES = [*(f"rand_both{x}" for x in ("", "_s1", "_s2")), "head_both", "rand_kg", "head_kg",
                  "rand_text", "head_text", *LEVELS]


def mode_spec(mode: str) -> dict:
    """base / kg / text / both / rand_* / head_* / both_l* -> cách nén."""
    sel, seed, part, levels = "rel", 0, mode, (KG_KEEP, SENT_KEEP, KG_ITEM_CHARS)
    if mode in LEVELS:
        part, levels = "both", LEVELS[mode]
    elif mode.startswith(("rand_", "head_")):
        sel, rest = mode.split("_", 1)
        part, _, sd = rest.partition("_s")
        seed = int(sd) if sd else 0
    kg_on, text_on = MODES[part] if part in MODES else PARTS[part]
    return {"kg": kg_on, "text": text_on, "sel": sel, "seed": seed, "kg_keep": levels[0], "sent_keep": levels[1],
            "item_chars": levels[2]}


def order(sims: np.ndarray, sel: str, rng: np.random.Generator) -> np.ndarray:
    """Thứ tự ưu tiên giữ lại: theo độ liên quan (rel), ngẫu nhiên (rand) hoặc theo vị trí gốc (head)."""
    if sel == "rel":
        return np.argsort(-sims)
    return rng.permutation(len(sims)) if sel == "rand" else np.arange(len(sims))
KG_ITEM_CHARS = 110       # độ dài mô tả tối đa mỗi entity / quan hệ sau khi nén (gốc: 220)
KG_KEEP = 0.5             # tỉ lệ entity / quan hệ giữ lại (gần câu hỏi nhất)
# Mức nén chọn bằng --calibrate CHỈ theo độ dài context (chưa chạy LLM / judge): giảm ký tự trung bình kg ~24%,
# text ~23%, both ~47% -> trải đều khoảng 19–53% của đề bài.
SENT_KEEP = 0.5           # tỉ lệ số câu giữ lại mỗi đoạn văn
KG_SHOWN_CHARS = 220      # rag_graphrag.KnowledgeGraph.kg_text hiển thị 220 ký tự đầu của mô tả


# ---------------- tiện ích ----------------
def unit(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


_EMB: dict[str, np.ndarray] = {}  # cache vector câu / mảnh mô tả (search_document)


def embed_docs(texts: list[str]) -> np.ndarray:
    todo = [t for t in dict.fromkeys(texts) if t not in _EMB]
    if todo:  # embed theo lô rồi đưa vào cache
        for t, v in zip(todo, get_embeddings().embed_documents(todo)):
            _EMB[t] = unit(v)
    return np.stack([_EMB[t] for t in texts])


def query_vec(question: str) -> np.ndarray:
    return unit(get_embeddings().embed_query(question))


# ---------------- nén KG ----------------
def shown_fragments(descriptions: list[str]) -> list[str]:
    """Các mảnh mô tả mà context gốc thật sự hiển thị (220 ký tự đầu), bỏ mảnh trùng / nằm trong mảnh khác."""
    frags = [f.strip() for f in "; ".join(descriptions)[:KG_SHOWN_CHARS].split("; ") if f.strip()]
    out = []
    for f in frags:
        low = f.lower().rstrip(".")
        if not any(low in o.lower() for o in out):
            out = [o for o in out if o.lower().rstrip(".") not in low] + [f]
    return out


def pick(frags: list[str], q: np.ndarray, budget: int, sel: str = "rel", rng=None) -> str:
    """Mảnh gần câu hỏi nhất trước, thêm mảnh tiếp theo nếu còn trong giới hạn; giữ thứ tự gốc."""
    sims = embed_docs(frags) @ q if sel == "rel" else np.zeros(len(frags))
    keep, used = [], 0
    for i in order(sims, sel, rng):
        if not keep or used + len(frags[i]) + 2 <= budget:
            keep.append(int(i))
            used += len(frags[i]) + 2
    return "; ".join(frags[i] for i in sorted(keep))


def top_items(lines: list[str], q: np.ndarray, keep: float = None, sel: str = "rel", rng=None) -> list[str]:
    """Giữ tỉ lệ KG_KEEP dòng (entity hoặc quan hệ) gần câu hỏi nhất, theo thứ tự gốc."""
    keep = KG_KEEP if keep is None else keep
    n = max(1, round(len(lines) * keep))
    sims = embed_docs(lines) @ q if sel == "rel" else np.zeros(len(lines))
    top = order(sims, sel, rng)[:n]
    return [lines[i] for i in sorted(int(i) for i in top)]


def kg_text_compressed(kg, hit: dict, q: np.ndarray, spec: dict | None = None, rng=None) -> str:
    spec = spec or mode_spec("kg")
    sel, chars = spec["sel"], spec["item_chars"]
    ents = [f"- {e['name']} ({e['type']}): {pick(shown_fragments(e['descriptions']), q, chars, sel, rng)}"
            for e in hit["entities"]]
    rels = [f"- {kg.entities[r['source']]['name']} -- {kg.entities[r['target']]['name']}: "
            f"{pick(shown_fragments(r['descriptions']), q, chars, sel, rng)}" for r in hit["relations"]]
    return ("Knowledge graph entities:\n" + "\n".join(top_items(ents, q, spec["kg_keep"], sel, rng))
            + "\n\nKnowledge graph relations:\n" + "\n".join(top_items(rels, q, spec["kg_keep"], sel, rng)))


# ---------------- nén văn bản ----------------
NOISE = [re.compile(p) for p in (
    r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]",          # đánh số trích dẫn [12], [1, 2], [40–42]
    r"https?://\S+|www\.\S+",                        # URL
    r"\S+@\S+\.\w+",                                 # email
    r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?(?:\s*\[[^\]]+\])?(?:\s*\d{1,2}\s\w{3}\s\d{4})?")]  # dòng arXiv


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # từ bị ngắt dòng có gạch nối
    for p in NOISE:
        text = p.sub("", text)
    text = re.sub(r"[ \t]+", " ", re.sub(r"(?<!\n)\n(?!\n)", " ", text))
    return re.sub(r" +([,.;:])", r"\1", text).strip()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“•])|\s*•\s*|\n{2,}", text)
    out = []
    for s in (p.strip() for p in parts if p and p.strip()):
        if out and len(s) < 25:   # mảnh quá ngắn (tiêu đề, số trang) gộp vào câu trước
            out[-1] += " " + s
        else:
            out.append(s)
    return out


def compress_passage(text: str, q: np.ndarray, keep: float = None, sel: str = "rel", rng=None) -> str:
    keep = SENT_KEEP if keep is None else keep
    sents = sentences(clean(text))
    if len(sents) <= 1:
        return " ".join(sents)
    n = max(1, round(len(sents) * keep))
    rel = sorted(int(i) for i in np.argsort(-(embed_docs(sents) @ q))[:n])
    if sel == "rel":
        return " ".join(sents[i] for i in rel)
    # baseline: cùng ngân sách ký tự với bản chọn theo độ liên quan (không chỉ cùng số câu: câu dài ngắn khác nhau)
    budget, top, used = sum(len(sents[i]) + 1 for i in rel), [], 0
    for i in order(np.zeros(len(sents)), sel, rng):
        if top and used + len(sents[i]) / 2 > budget:
            break
        top.append(int(i))
        used += len(sents[i]) + 1
    return " ".join(sents[i] for i in sorted(top))


# ---------------- context + trả lời ----------------
def build(question: str, mode: str) -> dict:
    spec = mode_spec(mode)
    kg_on, text_on = spec["kg"], spec["text"]
    kg = get_kg()
    hit = kg.search(question)
    q = query_vec(question) if (kg_on or text_on) else None
    # seed theo câu hỏi + seed của chế độ: cùng câu hỏi luôn được nén ngẫu nhiên giống nhau
    rng = np.random.default_rng(int(hashlib.sha256(f"{spec['seed']}:{question}".encode()).hexdigest()[:12], 16))
    kg_text = kg_text_compressed(kg, hit, q, spec, rng) if kg_on else kg.kg_text(hit)
    chunks = [{**c, "text": compress_passage(c["text"], q, spec["sent_keep"], spec["sel"], rng) if text_on
               else c["text"]} for c in hit["chunks"]]
    return {"kg_text": kg_text, "chunks": chunks, "context": format_context(kg_text, chunks),
            "hit": hit}


@lru_cache
def _counter():
    """LLM sinh câu trả lời với num_predict = 1: chỉ để lấy prompt_eval_count (số token của prompt)."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, temperature=0, seed=config.SEED,
                      num_ctx=config.NUM_CTX, num_predict=1)


def prompt_tokens(context: str, question: str) -> int:
    msg = (GENERATE_PROMPT | _counter()).invoke({"context": context, "question": question})
    return msg.response_metadata["prompt_eval_count"]


def answer(question: str, mode: str = "both") -> dict:
    t0 = time.perf_counter()
    b = build(question, mode)
    msg = (GENERATE_PROMPT | get_llm()).invoke({"context": b["context"], "question": question})
    latency = round(time.perf_counter() - t0, 2)
    tok = msg.response_metadata.get("prompt_eval_count")
    return {
        "answer": StrOutputParser().invoke(msg).strip(),
        "mode": mode,
        "contexts": [{k: c[k] for k in ("text", "source", "topic", "page", "chunk_id")} for c in b["chunks"]],
        "kg_context": b["kg_text"],
        "prompt_tokens": tok,
        "prompt_tokens_no_context": prompt_tokens("", question),   # để tính số token riêng của context
        "context_chars": len(b["context"]),
        "latency_s": latency,
        "llm_calls": 1,
        "trace": ["match_entities", "expand_relations", "select_chunks", *(["compress"] if mode != "base" else []),
                  "generate"],
    }


# ---------------- hiệu chỉnh mức nén (chỉ đo độ dài, không nhìn kết quả trả lời) ----------------
def calibrate():
    global KG_ITEM_CHARS, SENT_KEEP, KG_KEEP
    from evaluate import load_testset
    from evaluate_graphrag import TESTSET_FILES, gold_sources
    from kg_build import load_papers
    papers = set(load_papers(config.KG_K))
    qs = [t["question"] for t in load_testset([f for f in TESTSET_FILES if f.exists()])
          if t["type"] == "unanswerable" or set(gold_sources(t)) <= papers]
    base = {q: build(q, "base") for q in qs}
    print(f"{len(qs)} câu; context gốc trung bình {np.mean([len(b['context']) for b in base.values()]):.0f} ký tự "
          f"(mô tả KG {np.mean([len(b['kg_text']) for b in base.values()]):.0f})")
    for kg_keep, keep in ((1.0, 0.5), (0.67, 0.5), (0.5, 0.5), (0.5, 0.4), (0.67, 0.6)):
        KG_KEEP, SENT_KEEP = kg_keep, keep
        red = {m: [] for m in ("kg", "text", "both")}
        for q in qs:
            for m in red:
                red[m].append(1 - len(build(q, m)["context"]) / len(base[q]["context"]))
        print(f"KG_KEEP={kg_keep:.2f} KG_ITEM_CHARS={KG_ITEM_CHARS} SENT_KEEP={keep:.1f}: giảm ký tự " +
              ", ".join(f"{m} {np.mean(v):.1%} [{np.min(v):.0%}–{np.max(v):.0%}]" for m, v in red.items()))
    KG_KEEP, SENT_KEEP = 0.5, 0.5
    # mức nén thêm cho chế độ nén cả hai và các baseline: chỉ đo độ dài
    red = {m: [1 - len(build(q, m)["context"]) / len(base[q]["context"]) for q in qs] for m in BASELINE_MODES}
    for m, v in red.items():
        print(f"{m}: giảm ký tự {np.mean(v):.1%} [{np.min(v):.0%}–{np.max(v):.0%}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--mode", default="both", choices=[*MODES, *BASELINE_MODES])
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()
        sys.exit()
    q = " ".join(a.question) or "Which signature schemes did NIST select for standardization?"
    b0, b = build(q, "base"), build(q, a.mode)
    print(b["context"])
    print(f"\n--- {a.mode}: {len(b['context'])} / {len(b0['context'])} ký tự "
          f"(giảm {1 - len(b['context']) / len(b0['context']):.1%})")
