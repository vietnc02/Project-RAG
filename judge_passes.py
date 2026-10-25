"""Gộp 3 lần chấm của judge và đo độ lệch giữa các lần.

    python judge_passes.py results             # mọi scores*.p<N>.jsonl trong cây thư mục -> scores*.jsonl

Mỗi lần chấm (config.JUDGE_PASS = 0, 1, 2; cùng cách chấm) ghi scores_<hệ>.p<N>.jsonl. Bản gộp scores_<hệ>.jsonl (file mà
các bước tổng hợp / báo cáo đọc):
  - nhãn nhị phân (hallucination*, answer_correctness, refusal, retrieval_hit) và verdict từng đoạn (useful): đa số;
    khi mới có 2 lần chấm mà hai lần khác nhau thì lấy lần 0
  - faithfulness và các điểm liên tục (context_precision, context_recall, answer_relevancy, ...): trung bình các lần
  - danh sách claim, các trường khác: của lần 0
Câu chưa đủ 3 lần chấm dùng các lần đã có.
Độ nhất quán giữa các lượt ghi ra judge_passes.json cạnh mỗi nhóm file: tỉ lệ câu có điểm trùng ở mọi lượt, sai khác
tuyệt đối trung bình giữa hai lượt, độ lệch chuẩn giữa các lượt (trung bình theo câu).
"""
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

BINARY = {"hallucination", "hallucination_used", "hallucination_text", "hallucination_full", "answer_correctness",
          "refusal", "retrieval_hit"}
PASS_RE = re.compile(r"^(?P<base>.+)\.p(?P<n>\d+)\.jsonl$")


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in map(json.loads, f) if d}


def vote(vals: list):
    """Đa số; hoà (số lần chẵn) thì lấy giá trị của lần đầu tiên có giá trị."""
    v = [x for x in vals if not isnan(x)]
    if not v:
        return math.nan
    ones = sum(1 for x in v if x)
    if 2 * ones == len(v):
        return v[0]
    return int(2 * ones > len(v))


def mean(vals: list):
    v = [x for x in vals if not isnan(x)]
    return float(np.mean(v)) if v else math.nan


def merge(recs: list[dict]) -> dict:
    out = dict(recs[0])
    for k in set().union(*recs):
        vals = [r.get(k) for r in recs]
        if k == "useful":  # verdict từng đoạn: đa số theo vị trí
            n = max(len(v) for v in vals if isinstance(v, list))
            out[k] = [vote([v[i] if isinstance(v, list) and i < len(v) else None for v in vals]) for i in range(n)]
            out[k] = [None if isnan(x) else x for x in out[k]]
        elif k in BINARY:
            out[k] = vote(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) or v is None for v in vals) \
                and k not in ("id",):
            out[k] = mean(vals)
    out["n_passes"] = len(recs)
    return out


def agreement(per_pass: list[dict], ids: list[str]) -> dict:
    keys = sorted({k for p in per_pass for r in p.values() for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("n_passes",)})
    res = {}
    for k in keys:
        same, diffs, sds, n = 0, [], [], 0
        for i in ids:
            vals = [p[i].get(k) for p in per_pass]
            if any(isnan(v) for v in vals):
                continue
            n += 1
            same += all(abs(v - vals[0]) < 1e-9 for v in vals)
            diffs += [abs(a - b) for a, b in combinations(vals, 2)]
            sds.append(float(np.std(vals)))
        if n:
            res[k] = {"n": n, "all_equal": same / n, "mean_abs_diff": float(np.mean(diffs)), "mean_sd": float(np.mean(sds))}
    return res


def main(root: Path):
    groups = defaultdict(dict)
    for fp in root.rglob("*.jsonl"):
        m = PASS_RE.match(fp.name)
        if m and m["base"].startswith("scores"):
            groups[fp.parent / f"{m['base']}.jsonl"][int(m["n"])] = fp
    for canon, passes in sorted(groups.items()):
        per_pass = [load(passes[n]) for n in sorted(passes)]
        ids = [i for i in per_pass[0] if all(i in p for p in per_pass)]   # câu có đủ mọi lần chấm
        # mỗi câu: trung bình các lần chấm đã có (lần sau chưa chấm xong thì câu đó tạm dùng các lần đã có)
        merged = [merge([p[i] for p in per_pass if i in p]) for i in per_pass[0]]
        with open(canon, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats_path = canon.parent / "judge_passes.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        stats[canon.name] = {"passes": sorted(passes), "n_ids": len(ids),
                             "agreement": agreement(per_pass, ids) if len(per_pass) > 1 else {}}
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{canon.relative_to(root)}: {len(passes)} lượt, {len(ids)} câu")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
