"""Giai đoạn 2, việc 4 (Y3.2, Y3.5): kiểm tra tay bộ chuẩn và phép khớp cạnh của nội dung 3, fact retention của KG.

    python kg_audit.py --make            # phiếu testset/kg_audit/phieu_cham.html: duyệt lại 20% gold set + audit cạnh
    python kg_audit.py --score           #   KG10 / KG20 -> results/kg_scale/kg_audit.{md,json}, kg_relation_audit_k{10,20}.csv
    python kg_audit.py --facts           # fact retention (gọi LLM, chạy tiếp được): sự kiện + verdict theo KG10/20/30
    python kg_audit.py --make-facts      # phiếu testset/kg_audit/phieu_fact.html: kiểm tra tay 60 cặp (sự kiện, verdict)
    python kg_audit.py --score-facts     # -> results/kg_scale/fact_retention/fact_retention.{md,json}

1. Duyệt lại gold set (Y3.5): 20% câu hỏi (12/62, seed 42). Với mỗi câu: thực thể / quan hệ của bộ chuẩn
   (testset/kg_gold.json) và thực thể / quan hệ judge đề xuất nhưng lần duyệt đầu đã bỏ (testset/kg_gold_raw.json; bỏ qua
   mục chỉ bị đổi tên, tức có chung từ đặc trưng với một mục của bộ chuẩn), xáo trộn, không ghi mục nào đã được giữ.
   Người duyệt thứ hai chọn giữ / bỏ theo đúng quy tắc của lần duyệt đầu -> % khớp và Cohen's κ với lần duyệt đầu.
2. Audit cạnh (Y3.5): 40 quan hệ chuẩn có cạnh khớp trực tiếp cho mỗi KG10, KG20 (seed 42), như audit 40 cạnh KG30 đã
   có (kg_relation_audit.csv): mô tả cạnh có nói đúng quan hệ chuẩn không -> độ chính xác của phép khớp theo từng k.
3. Fact retention (Y3.2, theo bộ đo MINE của KGGen): mỗi bài trong 30 bài lấy 3 đoạn (seed 42), judge (llama3.1:8b, khác
   model dựng KG) tách 3 sự kiện mỗi đoạn; với từng KG, lấy phần KG theo đúng local search của GraphRAG (8 entity, 12
   quan hệ, không lấy passage) cho câu truy vấn là sự kiện, judge trả lời sự kiện có suy ra được từ phần KG đó không.
   Retention = tỉ lệ sự kiện suy ra được: trên sự kiện của các bài có trong KG và trên toàn bộ sự kiện. Kiểm tra tay
   60 cặp (30 yes, 30 no) để đo độ chính xác của judge và độ đúng của sự kiện.
"""
import argparse
import hashlib
import json
import random
import re

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

import config
from evaluate import bootstrap_ci, judge
from evaluate_kg_scale import KS, OUT
from human_judge import boot_agreement, cohen_kappa, landis_koch
from kg_build import load_chunks, load_papers
from kg_quality import GENERIC_WORDS, GOLD, GOLD_RAW, gold_key, words

DIR = config.PROJECT_DIR / "testset" / "kg_audit"
TEMPLATE = config.PROJECT_DIR / "testset" / "review" / "template.html"
SHEET, KEY, LABELS = DIR / "phieu_cham.html", DIR / "key.json", DIR / "human_labels.json"
FSHEET, FKEY, FLABELS = DIR / "phieu_fact.html", DIR / "key_fact.json", DIR / "human_labels_fact.json"
FDIR = OUT / "fact_retention"
FACTS, VERDICTS = FDIR / "facts.jsonl", FDIR / "verdicts.jsonl"
GOLD_FRACTION, N_EDGES, CHUNKS_PER_PAPER, N_FACT_CHECK = 0.2, 40, 3, 60
YN = [[1, "Có (1)", "y"], [0, "Không (0)", "n"]]
KEEP = [[1, "Giữ (1)", "y"], [0, "Bỏ (0)", "n"]]


def write_sheet(path, key_path, title, file, guide, cases, key):
    sheet_id = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()[:12]
    payload = json.dumps({"sheet_id": sheet_id, "title": title, "file": file, "guide": guide, "cases": cases},
                         ensure_ascii=False)
    DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload.replace("</", "<\\/"), 1),
                    encoding="utf-8")
    key_path.write_text(json.dumps({"seed": config.SEED, "sheet_id": sheet_id, "cases": key}, ensure_ascii=False,
                                   indent=1), encoding="utf-8")
    return sheet_id


def read_labels(path, key_path) -> tuple[dict, dict]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("sheet_id") != key["sheet_id"]:
        raise SystemExit(f"{path} không khớp phiếu hiện tại.")
    lab = {(x["code"], x["key"]): x["label"] for x in d["labels"]}
    if any(v is None for v in lab.values()):
        raise SystemExit(f"Còn {sum(v is None for v in lab.values())} ô chưa chấm.")
    return key, lab


# ---------------- 1 + 2. Duyệt lại gold set, audit cạnh KG10 / KG20 ----------------
GUIDE = [
    ("Mục đích", "Kiểm tra tay bộ chuẩn và phép khớp cạnh của nội dung 3. Phần G: duyệt lại bộ chuẩn của 12 câu hỏi (20%); "
                 "phần E: kiểm tra 80 cạnh KG khớp với quan hệ chuẩn (40 của KG10, 40 của KG20)."),
    ("G · Thực thể", "Giữ nếu đây là thực thể kỹ thuật cần có trong KG để trả lời câu hỏi theo ground truth (thuật toán, "
                     "giao thức, tấn công, thư viện, chuẩn, khái niệm...). Bỏ nếu là từ chung chung (\"data\", "
                     "\"participants\"), con số (\"60%\"), tên người, hoặc không cần để trả lời."),
    ("G · Quan hệ", "Giữ nếu ground truth nêu đúng quan hệ này giữa hai thực thể. Bỏ nếu quan hệ do suy diễn, không có "
                    "trong ground truth, hoặc nối tới con số / tên người."),
    ("E · Cạnh", "Có nếu ít nhất một mô tả cạnh KG nói đúng quan hệ chuẩn (cùng ý, có thể khác chữ). Không nếu các mô tả "
                 "chỉ nói một quan hệ khác giữa cùng hai thực thể (vd. chuẩn: \"OpenSSL dự định tích hợp Dilithium\", "
                 "cạnh: \"OpenSSL hỗ trợ Dilithium\")."),
    ("Khi xong", "Bấm \"Xuất JSON\", lưu vào testset/kg_audit/human_labels.json, rồi chạy python kg_audit.py --score."),
    ("Ước lượng thời gian", "Khoảng 1,5–2 giờ."),
]


def renamed(name: str, kept: list[str]) -> bool:
    """Mục bị bỏ ở lần duyệt đầu nhưng thực ra chỉ đổi tên: chung ít nhất 1 từ đặc trưng với một mục được giữ."""
    w = words(name) - GENERIC_WORDS
    return bool(w) and any(w & (words(k) - GENERIC_WORDS) for k in kept)


def make():
    if LABELS.exists():
        raise SystemExit(f"{LABELS} đã có (đã chấm).")
    rng = random.Random(config.SEED)
    gold = {g["id"]: g for g in json.loads(GOLD.read_text(encoding="utf-8"))}
    raw = {g["id"]: g for g in json.loads(GOLD_RAW.read_text(encoding="utf-8"))}
    qids = sorted(gold)
    sample = sorted(rng.sample(qids, round(GOLD_FRACTION * len(qids))))
    cases, key = [], {}
    for n, qid in enumerate(sample, 1):
        g, r = gold[qid], raw.get(qid, {"entities": [], "relations": []})
        ents = list(dict.fromkeys(g["entities"] + [x for rel in g["relations"] for x in rel[:2]]))
        kept_keys = {gold_key(e) for e in ents}
        cand_e = [(e, 1) for e in ents] + [(e, 0) for e in dict.fromkeys(r["entities"])
                                           if gold_key(e) not in kept_keys and not renamed(e, ents)]
        pairs = {tuple(sorted((gold_key(a), gold_key(b)))) for a, b, _ in g["relations"]}
        cand_r = [(rel, 1) for rel in g["relations"]] + [
            (rel, 0) for rel in r["relations"] if tuple(sorted((gold_key(rel[0]), gold_key(rel[1])))) not in pairs
            and not (renamed(rel[0], ents) and renamed(rel[1], ents))]
        rng.shuffle(cand_e)
        rng.shuffle(cand_r)
        code = f"G{n:02d}"
        key[code] = {"part": "gold", "id": qid, "entities": [[e, v] for e, v in cand_e],
                     "relations": [[rel, v] for rel, v in cand_r]}
        q = next(x for x in (raw.get(qid), g) if x)
        blocks = [{"title": "Câu hỏi", "text": q["question"]},
                  {"title": "Ground truth", "text": raw.get(qid, {}).get("ground_truth", "")}]
        blocks += [{"key": f"e:{i}", "group": "Thực thể: giữ trong bộ chuẩn?", "text": e, "prompt": "Giữ?",
                    "options": KEEP} for i, (e, _) in enumerate(cand_e)]
        blocks += [{"key": f"r:{i}", "group": "Quan hệ: ground truth có nêu?", "text": f"{a}  ⟷  {b}: {d}",
                    "prompt": "Giữ?", "options": KEEP} for i, ((a, b, d), _) in enumerate(cand_r)]
        cases.append({"code": code, "title": f"bộ chuẩn, câu {qid}", "blocks": blocks})
    rel = pd.read_csv(OUT / "kg_relations_per_question.csv")
    for k in (10, 20):
        d = rel[(rel.k == k) & (rel.direct == 1)].drop_duplicates(subset=["a", "b", "relation"])
        rows = d.sample(n=min(N_EDGES, len(d)), random_state=config.SEED).to_dict("records")
        for n, row in enumerate(rows, 1):
            code = f"E{k}-{n:02d}"
            key[code] = {"part": f"edge{k}", "k": k, **{c: row[c] for c in ("id", "a", "b", "relation",
                                                                              "edge_descriptions")}}
            cases.append({"code": code, "title": f"cạnh KG{k}", "blocks": [
                {"title": "Quan hệ chuẩn", "text": f"{row['a']}  ⟷  {row['b']}\n{row['relation']}"},
                {"title": "Mô tả cạnh trong KG", "text": "\n".join(f"• {x}" for x in str(row["edge_descriptions"]).split(" || "))},
                {"key": "edge", "prompt": "Cạnh KG nói đúng quan hệ chuẩn?", "options": YN}]})
    write_sheet(SHEET, KEY, "Phiếu kiểm tra tay · nội dung 3 (bộ chuẩn, cạnh KG)", "human_labels.json", GUIDE,
                cases, key)
    n_lab = sum(len(b["entities"]) + len(b["relations"]) if b["part"] == "gold" else 1 for b in key.values())
    print(f"Đã tạo {SHEET} ({len(sample)} câu bộ chuẩn, {sum(c['part'] != 'gold' for c in key.values())} cạnh; "
          f"{n_lab} ô cần chấm)")


def score():
    key, lab = read_labels(LABELS, KEY)
    rows, edges = [], []
    for code, c in key["cases"].items():
        if c["part"] == "gold":
            for kind, lst in (("entity", c["entities"]), ("relation", c["relations"])):
                for i, (_, first) in enumerate(lst):
                    rows.append({"code": code, "kind": kind, "first": first, "human": lab[(code, f"{kind[0]}:{i}")]})
        else:
            edges.append({"k": c["k"], "no": len(edges) + 1, **{f: c[f] for f in ("id", "a", "b", "relation",
                                                                                   "edge_descriptions")},
                          "correct": lab[(code, "edge")], "note": ""})
    g = pd.DataFrame(rows)
    gold_agree = {}
    for kind, d in g.groupby("kind"):
        d = d.rename(columns={"first": "first_review"})
        a = boot_agreement(d, "first_review")
        a["kept_by_first_also_kept"] = round(float(d[d.first_review == 1].human.mean()), 3)
        a["dropped_by_first_also_dropped"] = round(float(1 - d[d.first_review == 0].human.mean()), 3) \
            if (d.first_review == 0).any() else None
        a["n_dropped_shown"] = int((d.first_review == 0).sum())
        a["level"] = landis_koch(a["kappa"])
        gold_agree[kind] = a
    e = pd.DataFrame(edges)
    edge_prec = {}
    for k, d in e.groupby("k"):
        m, lo, hi = bootstrap_ci(list(d.correct.astype(float)))
        edge_prec[int(k)] = {"n": len(d), "correct": int(d.correct.sum()), "precision": round(m, 4),
                             "ci95": [round(lo, 4), round(hi, 4)]}
        d.drop(columns="k").assign(no=range(1, len(d) + 1)).to_csv(OUT / f"kg_relation_audit_k{k}.csv", index=False,
                                                                     encoding="utf-8-sig")
    a30 = pd.read_csv(OUT / "kg_relation_audit.csv")
    m, lo, hi = bootstrap_ci(list(a30.correct.astype(float)))
    edge_prec[30] = {"n": len(a30), "correct": int(a30.correct.sum()), "precision": round(m, 4),
                     "ci95": [round(lo, 4), round(hi, 4)]}
    stats = {"gold_review": {"n_questions": int(g.code.nunique()), **gold_agree}, "edge_audit": edge_prec}
    (OUT / "kg_audit.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Kiểm tra tay bộ chuẩn và phép khớp cạnh (nội dung 3)\n",
             f"## Duyệt lại bộ chuẩn: {stats['gold_review']['n_questions']} câu hỏi (20%)\n"]
    for kind, a in gold_agree.items():
        lines.append(f"- {kind}: {a['n_labels']} mục ({a['n_dropped_shown']} mục lần đầu đã bỏ); khớp {a['agreement']:.1%} "
                     f"{a['agreement_ci95']}, κ = {a['kappa']:.3f} {a['kappa_ci95']} ({a['level']}); mục lần đầu giữ "
                     f"cũng được giữ {a['kept_by_first_also_kept']:.1%}")
    lines += ["", "## Độ chính xác phép khớp cạnh (mô tả cạnh nói đúng quan hệ chuẩn)\n"]
    lines += [f"- KG{k}: {v['correct']}/{v['n']} = {v['precision']:.1%} [{v['ci95'][0]:.1%}; {v['ci95'][1]:.1%}]"
              for k, v in sorted(edge_prec.items())]
    (OUT / "kg_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "kg_audit.md").read_text(encoding="utf-8"))


# ---------------- 3. Fact retention ----------------
FACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract exactly 3 distinct factual statements from the passage of a scientific paper.\n"
     "Rules:\n"
     "- Each fact is ONE self-contained sentence stating a single piece of technical information from the passage "
     "(what a method does, a property, a result, a relation between two concepts).\n"
     "- Name concrete methods, systems, attacks or concepts. NEVER say 'this paper', 'the authors', 'the passage', "
     "'we', 'Table X', 'Figure X'.\n"
     "- No citation numbers, author names, or facts not stated in the passage.\n"
     '- If the passage has no meaningful technical content, return {{"facts": []}}.\n'
     'Return JSON: {{"facts": ["...", "...", "..."]}}'),
    ("human", "Paper title: {title}\n\nPassage:\n{text}"),
])
INFER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether a fact is contained in information retrieved from a knowledge graph (entities with "
     "descriptions, and relations between entities). Answer \"yes\" only if the knowledge graph information states the "
     "fact or it can be directly inferred from it; otherwise answer \"no\". Do not use outside knowledge.\n"
     'Return JSON: {{"verdict": "yes" or "no"}}'),
    ("human", "Knowledge graph information:\n{kg}\n\nFact: {fact}"),
])


SELF_REF = re.compile(r"\bthis (paper|article|study|work|survey|review|passage)\b|\bthe authors?\b|\bwe\b|"
                      r"\bour\b|\b(table|figure|fig\.) ?\d", re.I)


def good_chunk(c: dict) -> bool:
    t = c["text"]
    letters = sum(ch.isalpha() for ch in t) / max(len(t), 1)
    return len(t) >= 600 and letters > 0.65 and not re.search(r"acknowledg|copyright ©|all rights reserved|references",
                                                               t[:200], re.I)


def run_facts():
    from rag_graphrag import get_kg
    FDIR.mkdir(parents=True, exist_ok=True)
    papers = load_papers(30)
    rng = random.Random(config.SEED)
    by_paper = {p: [] for p in papers}
    for c in load_chunks():
        if c["source"] in by_paper and good_chunk(c):
            by_paper[c["source"]].append(c)
    done = {(f["chunk_id"]) for f in map(json.loads, open(FACTS, encoding="utf-8"))} if FACTS.exists() else set()
    with open(FACTS, "a", encoding="utf-8") as out:
        for p in tqdm(papers, desc="Tách sự kiện"):
            for c in rng.sample(sorted(by_paper[p], key=lambda x: x["chunk_id"]), min(CHUNKS_PER_PAPER, len(by_paper[p]))):
                if c["chunk_id"] in done:
                    continue
                r = judge(FACT_PROMPT, title=p, text=c["text"])
                # bỏ sự kiện nói về chính bài báo (không phải nội dung kỹ thuật), như khi sinh câu hỏi
                facts = [f.strip() for f in r.get("facts", []) if isinstance(f, str) and f.strip()
                         and not SELF_REF.search(f)][:3]
                out.write(json.dumps({"chunk_id": c["chunk_id"], "source": p, "paper_rank": papers.index(p) + 1,
                                      "text": c["text"], "facts": facts}, ensure_ascii=False) + "\n")
                out.flush()
    facts = [dict(f, fid=f"{f['chunk_id']}#{i}", fact=x) for f in map(json.loads, open(FACTS, encoding="utf-8"))
             for i, x in enumerate(f["facts"])]
    done = {(v["fid"], v["k"]) for v in map(json.loads, open(VERDICTS, encoding="utf-8"))} if VERDICTS.exists() else set()
    with open(VERDICTS, "a", encoding="utf-8") as out:
        for k in KS:
            kg = get_kg(k)
            for f in tqdm(facts, desc=f"Verdict KG{k}"):
                if (f["fid"], k) in done:
                    continue
                text = kg.kg_text(kg.search(f["fact"]))
                r = judge(INFER_PROMPT, kg=text, fact=f["fact"])
                v = str(r.get("verdict", "")).strip().lower()
                out.write(json.dumps({"fid": f["fid"], "k": k, "in_scope": int(f["paper_rank"] <= k), "kg_text": text,
                                      "verdict": 1 if v.startswith("y") else 0 if v.startswith("n") else None},
                                     ensure_ascii=False) + "\n")
                out.flush()
    print(f"{len(facts)} sự kiện; verdict -> {VERDICTS}")


def load_fact_data() -> pd.DataFrame:
    facts = {f"{f['chunk_id']}#{i}": dict(f, fact=x) for f in map(json.loads, open(FACTS, encoding="utf-8"))
             for i, x in enumerate(f["facts"])}
    v = pd.DataFrame(map(json.loads, open(VERDICTS, encoding="utf-8")))
    for c in ("fact", "source", "paper_rank", "text"):
        v[c] = v.fid.map(lambda x: facts[x][c])
    return v


FGUIDE = [
    ("Mục đích", "Kiểm tra bước tự động của fact retention: (1) sự kiện do judge tách có đúng với đoạn văn gốc không; "
                 "(2) judge kết luận sự kiện có / không suy ra được từ phần KG truy xuất được có đúng không."),
    ("Sự kiện đúng với đoạn văn?", "Có nếu đoạn văn gốc nêu đúng sự kiện này (không thêm, không sai ý)."),
    ("Suy ra được từ KG?", "Có nếu các mô tả entity / quan hệ trong khung \"Thông tin KG\" nêu sự kiện hoặc suy ra trực "
                           "tiếp được; Không nếu thiếu hoặc chỉ liên quan. Không dùng hiểu biết bên ngoài, không dùng "
                           "đoạn văn gốc ở bước này."),
    ("Khi xong", "Bấm \"Xuất JSON\", lưu vào testset/kg_audit/human_labels_fact.json, rồi chạy "
                 "python kg_audit.py --score-facts."),
    ("Ước lượng thời gian", "Khoảng 1 giờ."),
]


def make_facts():
    if FLABELS.exists():
        raise SystemExit(f"{FLABELS} đã có (đã chấm).")
    v = load_fact_data()
    v = v[(v.in_scope == 1) & v.verdict.notna()]
    rng = np.random.default_rng(config.SEED)
    pick = pd.concat([d.sample(n=N_FACT_CHECK // 2, random_state=config.SEED) for _, d in v.groupby("verdict")])
    pick = pick.iloc[rng.permutation(len(pick))]
    cases, key = [], {}
    for n, r in enumerate(pick.itertuples(), 1):
        code = f"F{n:02d}"
        key[code] = {"fid": r.fid, "k": int(r.k), "verdict": int(r.verdict)}
        cases.append({"code": code, "title": "", "blocks": [
            {"title": "Sự kiện", "text": r.fact},
            {"title": f"Đoạn văn gốc ({r.source})", "text": re.sub(r"(?<!\n)\n(?!\n)", " ", r.text)},
            {"key": "fact_ok", "prompt": "Sự kiện đúng với đoạn văn gốc?", "options": YN},
            {"title": "Thông tin KG truy xuất được cho sự kiện này", "text": r.kg_text, "kind": "kg"},
            {"key": "inferable", "prompt": "Sự kiện suy ra được từ thông tin KG này?", "options": YN}]})
    write_sheet(FSHEET, FKEY, "Phiếu kiểm tra tay · fact retention (nội dung 3)", "human_labels_fact.json", FGUIDE,
                cases, key)
    print(f"Đã tạo {FSHEET} ({len(cases)} cặp, {2 * len(cases)} ô cần chấm)")


def score_facts():
    v = load_fact_data()
    n_facts = int(v.fid.nunique())
    ret = {}
    for k, d in v.groupby("k"):
        ret[int(k)] = {}
        for scope, dd in (("in_scope", d[d.in_scope == 1]), ("all", d)):
            x = dd.verdict.dropna().astype(float)
            m, lo, hi = bootstrap_ci(list(x))
            ret[int(k)][scope] = {"n": len(x), "retention": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    stats = {"n_chunks": int(v.drop_duplicates("fid").fid.str.split("#").str[0].nunique()), "n_facts": n_facts,
             "missing_verdicts": int(v.verdict.isna().sum()), "retention": ret}
    if FLABELS.exists():
        key, lab = read_labels(FLABELS, FKEY)
        rows = [{"code": c, "judge": x["verdict"], "human": lab[(c, "inferable")], "fact_ok": lab[(c, "fact_ok")]}
                for c, x in key["cases"].items()]
        h = pd.DataFrame(rows)
        a = boot_agreement(h, "judge")
        a["level"] = landis_koch(a["kappa"])
        a["judge_yes_precision"] = round(float(h[h.judge == 1].human.mean()), 3)
        a["judge_no_npv"] = round(float(1 - h[h.judge == 0].human.mean()), 3)
        stats["human_check"] = {**a, "fact_ok": round(float(h.fact_ok.mean()), 3), "n": len(h)}
        # retention hiệu chỉnh theo độ chính xác của verdict yes / no trên mẫu kiểm tra tay
        p1, p0 = a["judge_yes_precision"], 1 - a["judge_no_npv"]
        for k in ret:
            r = ret[k]["in_scope"]["retention"]
            ret[k]["in_scope"]["retention_adjusted"] = round(r * p1 + (1 - r) * p0, 4)
    (FDIR / "fact_retention.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Fact retention của KG10 / KG20 / KG30 (kiểu MINE, KGGen)\n",
             f"{stats['n_facts']} sự kiện từ {stats['n_chunks']} đoạn (3 đoạn mỗi bài, 30 bài); verdict thiếu: "
             f"{stats['missing_verdicts']}.\n"]
    for k, r in sorted(ret.items()):
        lines.append(f"- KG{k}: sự kiện của các bài trong KG {r['in_scope']['retention']:.1%} "
                     f"[{r['in_scope']['ci95'][0]:.1%}; {r['in_scope']['ci95'][1]:.1%}] (n = {r['in_scope']['n']})"
                     + (f", hiệu chỉnh {r['in_scope']['retention_adjusted']:.1%}" if 'retention_adjusted' in r['in_scope']
                        else "")
                     + f"; toàn bộ sự kiện {r['all']['retention']:.1%} (n = {r['all']['n']})")
    if "human_check" in stats:
        hc = stats["human_check"]
        lines += ["", f"Kiểm tra tay {hc['n']} cặp: sự kiện đúng với đoạn văn {hc['fact_ok']:.1%}; verdict của judge khớp "
                      f"người {hc['agreement']:.1%}, κ = {hc['kappa']:.3f} ({hc['level']}); verdict yes đúng "
                      f"{hc['judge_yes_precision']:.1%}, verdict no đúng {hc['judge_no_npv']:.1%}."]
    (FDIR / "fact_retention.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((FDIR / "fact_retention.md").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    for a in ("make", "score", "facts", "make-facts", "score-facts"):
        ap.add_argument(f"--{a}", action="store_true")
    args = ap.parse_args()
    if args.make:
        make()
    elif args.score:
        score()
    elif args.facts:
        run_facts()
    elif args.make_facts:
        make_facts()
    elif args.score_facts:
        score_facts()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
