"""Ý 3: chất lượng Knowledge Graph dựng từ k = 10 / 20 / 30 bài (entity coverage, relation completeness).

    python kg_quality.py --structure      # thống kê cấu trúc + đường cong tăng trưởng theo số bài (không cần LLM)
    python kg_quality.py --gold           # judge trích thực thể / quan hệ chuẩn của 62 câu có đáp án
                                          #   -> testset/kg_gold_raw.json (duyệt tay: testset/review_kg_gold.py)
    python kg_quality.py --coverage       # entity coverage + relation completeness của KG10 / KG20 / KG30
    python kg_quality.py --all            # --structure --coverage (bộ chuẩn testset/kg_gold.json có sẵn trong repo)
Kết quả: results/kg_scale/kg_quality.{json,md}, kg_growth.csv, kg_coverage.csv, kg_coverage_per_question.csv,
kg_relations_per_question.csv (từng quan hệ chuẩn + mô tả cạnh KG khớp được), entity_matches.jsonl.
Chỉ bước --gold gọi LLM; --structure / --coverage tất định, chạy lại cho đúng kết quả cũ.

KG k bài (data/kg/k<k>/) = KG dựng từ k bài đầu của data/kg/papers.json bằng cùng cache trích xuất của ý 2.

Bộ chuẩn (gold): với mỗi câu có đáp án, các thực thể kỹ thuật cần có trong KG để trả lời câu hỏi và các quan hệ
giữa chúng mà đáp án chuẩn nêu ra. Judge (llama3.1:8b, khác model dựng KG) trích từ câu hỏi + đáp án chuẩn, sau đó
duyệt tay (testset/review_kg_gold.py).

Hai chỉ số chính:
  - entity coverage       = #thực thể chuẩn có trong KG / #thực thể chuẩn
      khớp = trùng tên sau chuẩn hoá (như bước gộp entity của kg_build: chữ thường, số ít, tên đầy đủ <-> viết tắt)
      phụ: *_lenient (cận trên: thêm entity có tên chứa trọn cụm từ của thực thể chuẩn, vd. "dilithium-3")
  - relation completeness = #quan hệ chuẩn có cạnh trực tiếp trong KG nối 2 thực thể đã khớp / #quan hệ chuẩn
      phụ: *_lenient (đầu mút khớp lenient), *_2hop (2 thực thể nối nhau trong <= 2 bước)
Tính trên 3 tập câu hỏi (xem evaluate_kg_scale.py): trong phạm vi KG (bài nguồn thuộc k bài), S10 cố định, toàn bộ
62 câu. Mỗi thực thể / quan hệ chuẩn tính 1 lần trong một tập (gộp theo tên chuẩn hoá).
Chỉ số cấu trúc (không cần bộ chuẩn): số entity / quan hệ, tỉ lệ entity / quan hệ xuất hiện ở >= 2 bài (liên kết
giữa các bài), bậc trung bình, thành phần liên thông lớn nhất, số entity mới mỗi bài thêm vào (độ bão hoà).
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

import config
from evaluate import bootstrap_ci, judge
from evaluate_graphrag import gold_chunks
from evaluate_kg_scale import KS, OUT, load_items
from kg_build import load_papers, norm, split_acronym

GOLD_RAW = config.PROJECT_DIR / "testset" / "kg_gold_raw.json"
GOLD = config.PROJECT_DIR / "testset" / "kg_gold.json"
MATCHES = OUT / "entity_matches.jsonl"
N_ORDERS = 20          # số thứ tự bài ngẫu nhiên cho đường cong tăng trưởng


# ---------------- 1. Cấu trúc KG + đường cong tăng trưởng ----------------
def load_graph(k: int) -> dict:
    fp = config.KG_DIR / f"k{k}" / "graph.json"
    if not fp.exists():
        raise SystemExit(f"Chưa có {fp}: chạy python kg_build.py --build --k {k}")
    return json.loads(fp.read_text(encoding="utf-8"))


def largest_component(n: int, edges) -> int:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return max(Counter(find(i) for i in range(n)).values(), default=0)


def growth(order: list[str], g30: dict) -> list[dict]:
    """Chiếu KG30 xuống k bài đầu của `order`: entity / quan hệ có mặt khi có ít nhất 1 nguồn trong k bài đó.
    (Xấp xỉ KG dựng lại từ k bài: chỉ khác ở vài tên viết tắt được gộp khác đi; so với KG thật ở k = 10/20/30.)"""
    pos = {s: i for i, s in enumerate(order)}
    e_first = [sorted(pos[s] for s in e["sources"]) for e in g30["entities"]]
    r_first = [sorted(pos[s] for s in r["sources"]) for r in g30["relations"]]
    rows, prev = [], 0
    for k in range(1, len(order) + 1):
        ents = [i for i, p in enumerate(e_first) if p[0] < k]
        rels = [j for j, p in enumerate(r_first) if p[0] < k]
        idx = {e: n for n, e in enumerate(ents)}
        edges = [(idx[g30["relations"][j]["source"]], idx[g30["relations"][j]["target"]]) for j in rels]
        rows.append({"k": k, "entities": len(ents), "relations": len(rels), "new_entities": len(ents) - prev,
                     "entities_2plus_share": sum(len([x for x in e_first[i] if x < k]) >= 2 for i in ents) / len(ents),
                     "relations_2plus_share": (sum(len([x for x in r_first[j] if x < k]) >= 2 for j in rels)
                                               / max(len(rels), 1)),
                     "largest_component_share": largest_component(len(ents), edges) / len(ents)})
        prev = len(ents)
    return rows


def structure() -> dict:
    stats = {}
    for k in KS:
        s = load_graph(k)["stats"]
        n = max(s["entities"], 1)
        stats[k] = {**{c: s[c] for c in ("papers", "chunks", "entities", "relations", "entities_in_2plus_papers",
                                         "relations_in_2plus_papers", "isolated_entities", "avg_degree",
                                         "components", "largest_component")},
                    "entities_per_paper": round(s["entities"] / s["papers"], 1),
                    "entities_2plus_share": round(s["entities_in_2plus_papers"] / n, 4),
                    "relations_2plus_share": round(s["relations_in_2plus_papers"] / max(s["relations"], 1), 4),
                    "isolated_share": round(s["isolated_entities"] / n, 4),
                    "largest_component_share": round(s["largest_component"] / n, 4)}
    g30 = load_graph(30)
    order = load_papers(30)
    rows = [{"order": "papers.json", **r} for r in growth(order, g30)]
    rng = random.Random(config.SEED)
    for i in tqdm(range(N_ORDERS), desc="Đường cong tăng trưởng (thứ tự ngẫu nhiên)"):
        perm = order[:]
        rng.shuffle(perm)
        rows += [{"order": f"random{i + 1}", **r} for r in growth(perm, g30)]
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "kg_growth.csv", index=False, encoding="utf-8-sig")
    proj = curve[curve.order == "papers.json"].set_index("k")
    check = {k: {"entities_projected": int(proj.loc[k, "entities"]), "entities_built": stats[k]["entities"],
                 "relations_projected": int(proj.loc[k, "relations"]), "relations_built": stats[k]["relations"]}
             for k in KS}
    rnd = curve[curve.order != "papers.json"].groupby("k")["new_entities"].mean()
    return {"kg": stats, "projection_check": check,
            "new_entities_per_paper_random_orders": {f"{a}-{b}": round(float(rnd.loc[a:b].mean()), 1)
                                                     for a, b in ((2, 10), (11, 20), (21, 30))}}


# ---------------- 2. Bộ chuẩn: thực thể / quan hệ cần cho từng câu hỏi ----------------
GOLD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You annotate questions about scientific papers for evaluating a knowledge graph.\n"
     "Given a question and its reference answer, list:\n"
     "- entities: the specific technical entities that a knowledge graph must contain to answer the question "
     "(algorithms, protocols, attacks, defenses, systems/tools, concepts, metrics, datasets, organizations, "
     "standards, hardware), written as in the text; use 'Full Name (ACRONYM)' if both are given. Skip numbers, "
     "years, people, paper titles and generic words ('method', 'approach', 'paper', 'results', 'system').\n"
     "- relations: pairs of entities from your list that the reference answer explicitly relates, with a short "
     "phrase stating the relation.\n"
     'Return JSON: {{"entities": ["..."], "relations": [{{"source": "...", "target": "...", "relation": "..."}}]}}\n\n'
     "Example question: Which mechanism lets routers reject BGP hijacking announcements?\n"
     "Example reference answer: Resource Public Key Infrastructure (RPKI) lets routers validate route origins "
     "and reject announcements from BGP hijacking.\n"
     'Example output: {{"entities": ["Resource Public Key Infrastructure (RPKI)", "BGP hijacking", '
     '"route origin validation"], "relations": [{{"source": "Resource Public Key Infrastructure (RPKI)", '
     '"target": "BGP hijacking", "relation": "RPKI lets routers reject BGP hijacking announcements"}}, '
     '{{"source": "Resource Public Key Infrastructure (RPKI)", "target": "route origin validation", '
     '"relation": "RPKI is used to validate route origins"}}]}}'),
    ("human", "Question: {question}\nReference answer: {reference}"),
])


def make_gold():
    items = [t for t in load_items() if t["type"] != "unanswerable"]
    rows = []
    for t in tqdm(items, desc="Trích thực thể / quan hệ chuẩn"):
        r = judge(GOLD_PROMPT, question=t["question"], reference=t["ground_truth"])
        ents = [e.strip() for e in r.get("entities", []) if isinstance(e, str) and e.strip()]
        rels = [[x["source"].strip(), x["target"].strip(), str(x.get("relation", "")).strip()]
                for x in r.get("relations", []) if isinstance(x, dict)
                and isinstance(x.get("source"), str) and isinstance(x.get("target"), str)]
        rows.append({"id": t["id"], "type": t["type"], "question": t["question"], "ground_truth": t["ground_truth"],
                     "entities": ents, "relations": rels})
    GOLD_RAW.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} câu, {sum(len(r['entities']) for r in rows)} thực thể, "
          f"{sum(len(r['relations']) for r in rows)} quan hệ -> {GOLD_RAW}")


def load_gold() -> dict:
    if not GOLD.exists():
        raise SystemExit(f"Chưa có {GOLD}: chạy python kg_quality.py --gold rồi python testset/review_kg_gold.py")
    gold = {}
    for g in json.loads(GOLD.read_text(encoding="utf-8")):
        ents = list(dict.fromkeys(g["entities"] + [x for r in g["relations"] for x in r[:2]]))
        gold[g["id"]] = {"entities": ents, "relations": g["relations"]}
    return gold


def gold_key(name: str) -> str:
    return norm(split_acronym(name)[0])


# ---------------- 3. Khớp thực thể chuẩn với KG ----------------
# Khớp tất định (không dùng LLM): judge 7-8B local không phân biệt ổn định "cùng khái niệm" với khái niệm rộng /
# hẹp hơn (vd. nhận "Dilithium" = "dilithium-3", "FPGA" = "artix-7 fpga") nên không dùng để khớp.
#   strict (chính): trùng tên sau chuẩn hoá — cùng quy tắc gộp entity của kg_build (norm + tên đầy đủ / viết tắt),
#                   so với tên, viết tắt và key của entity trong KG
#   lenient (phụ, cận trên): thêm entity có tên chứa trọn các từ của thực thể chuẩn và chung ít nhất 1 từ đặc trưng
#                   (vd. "Faraday effect" ~ "faraday-effect based polarization rotation", "Dilithium" ~ "dilithium-3")
GENERIC_WORDS = {"network", "quantum", "attack", "algorithm", "system", "key", "data", "model", "security",
                 "protocol", "scheme", "computer", "computing", "cryptography", "circuit", "learning", "machine",
                 "detection", "based", "classical", "neural", "method", "technique", "encryption", "signature",
                 "of", "the", "and", "for", "in", "on", "with", "a", "an", "to", "via", "by"}


def words(name: str) -> frozenset:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")) else w
                     for w in re.split(r"[^a-z0-9+]+", norm(name)) if w)


class KGIndex:
    def __init__(self, k: int):
        g = load_graph(k)
        self.k, self.entities, self.relations = k, g["entities"], g["relations"]
        self.lookup = defaultdict(set)    # tên chuẩn hoá (tên, viết tắt, key) -> entity id
        for e in self.entities:
            for name in (e["key"], e["name"], *e["aliases"]):
                self.lookup[norm(name)].add(e["id"])
        self.words = [words(e["key"]) for e in self.entities]
        self.adj = defaultdict(set)
        self.edges = defaultdict(list)    # (id nhỏ, id lớn) -> quan hệ
        for r in self.relations:
            self.adj[r["source"]].add(r["target"])
            self.adj[r["target"]].add(r["source"])
            self.edges[tuple(sorted((r["source"], r["target"])))].append(r)
        self.chunk_ids = {c for x in (*self.entities, *self.relations) for c in x["chunk_ids"]}

    def strict(self, name: str) -> set[int]:
        full, short = split_acronym(name)
        return set().union(*(self.lookup.get(norm(x), set()) for x in (name, full, short) if x))

    def lenient(self, name: str) -> set[int]:
        out = self.strict(name)
        for x in filter(None, split_acronym(name)):
            g = words(x)
            if g - GENERIC_WORDS:
                out |= {i for i, w in enumerate(self.words) if g <= w}
        return out


@lru_cache
def kg_index(k: int) -> KGIndex:
    return KGIndex(k)


def within_2_hops(idx: KGIndex, src: set[int], dst: set[int]) -> bool:
    frontier = set(src)
    for _ in range(2):
        frontier |= {n for x in frontier for n in idx.adj[x]}
        if frontier & dst:
            return True
    return False


def coverage() -> dict:
    gold = load_gold()
    items = [t for t in load_items() if t["id"] in gold]
    matches, ent_rows, rel_rows, q_rows = [], [], [], []
    for k in KS:
        idx = kg_index(k)
        for n in sorted({n for g in gold.values() for n in g["entities"]}):
            s, le = idx.strict(n), idx.lenient(n)
            matches.append({"k": k, "gold": n, "strict": sorted(idx.entities[i]["key"] for i in s),
                            "lenient_only": sorted(idx.entities[i]["key"] for i in le - s)[:30],
                            "n_lenient": len(le)})
        for t in tqdm(items, desc=f"Độ phủ KG{k}"):
            g = gold[t["id"]]
            st = {n: idx.strict(n) for n in g["entities"]}
            le = {n: idx.lenient(n) for n in g["entities"]}
            for n in g["entities"]:
                ent_rows.append({"k": k, "id": t["id"], "gold_key": gold_key(n), "covered": int(bool(st[n])),
                                 "covered_lenient": int(bool(le[n]))})
            cnt = Counter()
            for a, b, desc in g["relations"]:
                row = {"k": k, "id": t["id"], "gold_key": " -- ".join(sorted((gold_key(a), gold_key(b)))),
                       "a": a, "b": b, "relation": desc}
                for tag, m in (("", st), ("_lenient", le)):
                    ea, eb = m[a], m[b]
                    edges = [r for x in ea for y in eb if x != y for r in idx.edges.get(tuple(sorted((x, y))), [])]
                    row[f"endpoints{tag}"] = int(bool(ea and eb))
                    row[f"direct{tag}"] = int(bool(edges))
                    row[f"hop2{tag}"] = int(bool(ea and eb) and (bool(edges) or within_2_hops(idx, ea, eb)))
                    if not tag:
                        row["edge_descriptions"] = " || ".join(sorted({d for r in edges for d in r["descriptions"]})[:5])
                    cnt.update({f"direct{tag}": row[f"direct{tag}"]})
                rel_rows.append(row)
            ne, nr = len(g["entities"]), len(g["relations"])
            cov = sum(bool(v) for v in st.values())
            q_rows.append({"k": k, "id": t["id"], "type": t["type"], "scope": t["scope"],
                           "in_scope": int(t["scope"] <= k), "n_entities": ne, "entities_covered": cov,
                           "entity_coverage": cov / ne,
                           "entity_coverage_lenient": sum(bool(v) for v in le.values()) / ne,
                           "n_relations": nr, "relations_direct": cnt["direct"],
                           "relation_completeness": cnt["direct"] / nr if nr else np.nan,
                           "relation_completeness_lenient": cnt["direct_lenient"] / nr if nr else np.nan,
                           "gold_chunk_reachable": float(np.mean([c in idx.chunk_ids for c in gold_chunks(t)]))})
    qdf, edf, rdf = pd.DataFrame(q_rows), pd.DataFrame(ent_rows), pd.DataFrame(rel_rows)
    qdf.to_csv(OUT / "kg_coverage_per_question.csv", index=False, encoding="utf-8-sig")
    rdf.to_csv(OUT / "kg_relations_per_question.csv", index=False, encoding="utf-8-sig")
    with open(MATCHES, "w", encoding="utf-8") as f:
        for m in matches:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def ci(x):
        m, lo, hi = bootstrap_ci(list(x))
        return round(m, 4), f"[{lo:.3f}, {hi:.3f}]"

    summary = []
    scope_of = {t["id"]: t["scope"] for t in items}
    for set_name in ("in_scope", "S10", "S20", "all"):
        for k in KS:
            if set_name == "S20" and k < 20:
                continue
            keep = {i for i, s in scope_of.items()
                    if {"in_scope": s <= k, "S10": s == 10, "S20": s <= 20, "all": True}[set_name]}
            e = edf[(edf.k == k) & edf.id.isin(keep)].groupby("gold_key")[["covered", "covered_lenient"]].max()
            r = rdf[(rdf.k == k) & rdf.id.isin(keep)].groupby("gold_key")[
                ["endpoints", "direct", "hop2", "direct_lenient"]].max()
            q = qdf[(qdf.k == k) & qdf.id.isin(keep)]
            row = {"set": set_name, "k": k, "n_questions": len(keep), "n_gold_entities": len(e),
                   "n_gold_relations": len(r)}
            row["entity_coverage"], row["entity_coverage_ci95"] = ci(e["covered"])
            row["relation_completeness"], row["relation_completeness_ci95"] = ci(r["direct"])
            row["entity_coverage_lenient"], _ = ci(e["covered_lenient"])
            row["relation_completeness_lenient"], _ = ci(r["direct_lenient"])
            row["relation_completeness_2hop"], _ = ci(r["hop2"])
            row["relation_endpoints_covered"], _ = ci(r["endpoints"])
            row["questions_all_entities_covered"] = round(float((q.entity_coverage >= 1).mean()), 4)
            row["gold_chunk_reachable"] = round(q.gold_chunk_reachable.mean(), 4)
            summary.append(row)
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "kg_coverage.csv", index=False, encoding="utf-8-sig")
    # Kiểm tra tay: 40 quan hệ chuẩn (khác nhau) ngẫu nhiên (seed 42) có cạnh khớp trong KG30, đọc mô tả cạnh xem
    # có nói đúng quan hệ chuẩn không -> độ chính xác của phép khớp "có cạnh trực tiếp" (kg_relation_audit.csv)
    audit = OUT / "kg_relation_audit.csv"
    extra = {}
    if audit.exists():
        a = pd.read_csv(audit)
        extra["relation_audit"] = {"n": len(a), "correct": int(a["correct"].sum())}
    return {**extra,"gold": {"questions": len(items), "entities": int(edf[edf.k == KS[-1]].gold_key.nunique()),
                     "relations": int(rdf[rdf.k == KS[-1]].gold_key.nunique())},
            "coverage": sm.to_dict(orient="records")}


# ---------------- 4. Báo cáo ----------------
SET_TITLES = {"in_scope": "trong phạm vi KG", "S10": "S10 (10 bài đầu)", "S20": "S20 (20 bài đầu)",
              "all": "toàn bộ 62 câu"}


def write_md(q: dict):
    lines = ["# Chất lượng KG theo số bài (k = 10 / 20 / 30)\n"]
    main = []
    if "coverage" in q:
        sm = pd.DataFrame(q["coverage"])
        sm["set"] = sm["set"].map(SET_TITLES)
        main += [f"Bộ chuẩn: {q['gold']['questions']} câu có đáp án, {q['gold']['entities']} thực thể, "
                 f"{q['gold']['relations']} quan hệ (trích từ câu hỏi + đáp án chuẩn, duyệt tay).\n",
                 sm[["set", "k", "n_questions", "n_gold_entities", "entity_coverage", "entity_coverage_ci95",
                     "n_gold_relations", "relation_completeness", "relation_completeness_ci95"]]
                 .to_markdown(index=False), ""]
    if "kg" in q:
        st = pd.DataFrame(q["kg"]).T.rename_axis("k").reset_index()
        main += ["Cấu trúc KG:\n", st[["k", "papers", "chunks", "entities", "relations", "entities_per_paper",
                                       "entities_2plus_share", "relations_2plus_share", "avg_degree",
                                       "largest_component_share", "isolated_share"]].to_markdown(index=False), ""]
    lines += ["<!-- main -->", *main, "<!-- main -->", ""]
    if "coverage" in q:
        sm = pd.DataFrame(q["coverage"])
        lines += ["## Chỉ số phụ\n",
                  sm[["set", "k", "entity_coverage_lenient", "relation_completeness_lenient",
                      "relation_completeness_2hop", "relation_endpoints_covered", "questions_all_entities_covered",
                      "gold_chunk_reachable"]].to_markdown(index=False), ""]
        if "relation_audit" in q:
            a = q["relation_audit"]
            lines += [f"Kiểm tra tay {a['n']} cạnh KG30 khớp với quan hệ chuẩn (`kg_relation_audit.csv`): "
                      f"{a['correct']}/{a['n']} cạnh nói đúng quan hệ chuẩn ({a['correct'] / a['n']:.0%}).\n"]
    if "kg" in q:
        lines += ["## Đường cong tăng trưởng (`kg_growth.csv`)\n",
                  f"Số entity mới trung bình mỗi bài thêm vào ({N_ORDERS} thứ tự bài ngẫu nhiên): "
                  f"`{json.dumps(q['new_entities_per_paper_random_orders'])}`\n",
                  "Kiểm tra phép chiếu KG30 -> k bài (dùng cho đường cong) so với KG dựng thật: "
                  f"`{json.dumps(q['projection_check'])}`\n"]
    lines += ["## Định nghĩa\n",
              "- **entity coverage**: tỉ lệ thực thể chuẩn có trong KG (trùng tên sau chuẩn hoá, cùng quy tắc gộp "
              "entity của `kg_build`). `_lenient` (cận trên): thêm entity có tên chứa trọn cụm từ của thực thể chuẩn.",
              "- **relation completeness**: tỉ lệ quan hệ chuẩn có cạnh trực tiếp trong KG giữa 2 thực thể đã khớp; "
              "`_lenient`: đầu mút khớp lenient; `_2hop`: nối nhau trong <= 2 bước; `relation_endpoints_covered`: "
              "cả 2 đầu mút có trong KG.",
              "- **gold_chunk_reachable**: tỉ lệ đoạn gốc của câu hỏi có ít nhất 1 entity/quan hệ trong KG "
              "(GraphRAG chỉ lấy được đoạn văn qua entity/quan hệ).",
              "- `*_2plus_share`: tỉ lệ entity / quan hệ xuất hiện ở >= 2 bài (liên kết giữa các bài)."]
    (OUT / "kg_quality.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "kg_quality.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not (args.structure or args.gold or args.coverage or args.all):
        ap.print_help()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "kg_quality.json"
    q = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    if args.gold:
        make_gold()
    if args.structure or args.all:
        q.update(structure())
    if args.coverage or args.all:
        q.update(coverage())
    if q:
        fp.write_text(json.dumps(q, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_md(q)


if __name__ == "__main__":
    main()
