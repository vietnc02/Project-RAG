"""Thống kê dùng chung cho make_figures*.py (nội dung 1–4).

- Mỗi lần bootstrap tạo bộ sinh số mới từ config.SEED, nên cùng dữ liệu luôn cho cùng CI, bất kể thứ tự tính trong
  script (trước đây mỗi script dùng chung một bộ sinh số cho mọi bảng, cùng dữ liệu tính ở hai chỗ ra CI hơi khác).
- holm(): hiệu chỉnh đa so sánh Holm (kiểm soát family-wise error rate) cho một "họ" so sánh = mọi paired difference
  trong một bảng. Kết luận "có statistical significance" chỉ dùng khi p đã hiệu chỉnh < 0,05; chênh lệch có CI 95%
  không chứa 0 nhưng không qua hiệu chỉnh được ghi là "gợi ý".
- outcomes(): xếp mỗi câu vào một nhóm kết cục, tách hallucination khỏi refusal.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

N_BOOT, N_BOOT_PAIRED, ALPHA = 2000, 5000, 0.05


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def rng():
    return np.random.default_rng(config.SEED)


def boot(x, n=N_BOOT):
    """Trung bình + CI 95% bootstrap: [mean, lo, hi] (NaN nếu không có giá trị)."""
    x = np.asarray([v for v in x if not isnan(v)], float)
    if len(x) == 0:
        return [math.nan] * 3
    means = rng().choice(x, size=(n, len(x))).mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(x.mean()), float(lo), float(hi)]


def paired_boot(d, n=N_BOOT_PAIRED):
    """Paired difference: trung bình của d, CI 95% bootstrap và p hai phía (None nếu không có cặp)."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return None
    bs = rng().choice(d, size=(n, len(d))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 1.0 if not d.any() else float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d), "p": p}


def holm(entries):
    """Thêm p_holm, significant (CI không chứa 0) và significant_holm (p_holm < ALPHA) vào từng dict của một họ so
    sánh. Bỏ qua None và các so sánh không có khác biệt nào (d toàn 0). Trả về số so sánh trong họ."""
    es = [e for e in entries if e is not None and not (e["delta"] == 0 and e["lo"] == 0 and e["hi"] == 0)]
    m = len(es)
    running = 0.0
    for rank, e in enumerate(sorted(es, key=lambda e: e["p"])):
        running = max(running, min(1.0, (m - rank) * e["p"]))
        e["p_holm"] = running
    for e in entries:
        if e is None:
            continue
        e.setdefault("p_holm", 1.0)
        e["family_size"] = m
        e["significant"] = bool(e["lo"] > 0 or e["hi"] < 0)
        e["significant_holm"] = bool(e["significant"] and e["p_holm"] < ALPHA)
    return m


OUTCOMES = ["grounded_correct", "correct_unsupported", "wrong", "refused_wrongly",
            "refused_correctly", "answered_out_of_corpus"]


def outcome(answerable, refusal, correct, hallucination):
    """Nhóm kết cục của một câu:
    câu có đáp án: đúng và mọi claim có căn cứ / đúng nhưng có claim không được hỗ trợ (thường do kiến thức tham số) /
    sai / refuse oan; câu out-of-corpus: refuse đúng / vẫn trả lời."""
    if not answerable:
        return "refused_correctly" if refusal else "answered_out_of_corpus"
    if refusal:
        return "refused_wrongly"
    if correct:
        return "correct_unsupported" if hallucination else "grounded_correct"
    return "wrong"


def outcomes(rows):
    """rows: các bộ (answerable, refusal, correct, hallucination) -> {nhóm: số câu} và các tỉ lệ tách refusal:
    hallucination và correctness trên câu có trả lời, refusal đúng / oan."""
    cnt = {k: 0 for k in OUTCOMES}
    for r in rows:
        cnt[outcome(*r)] += 1
    ans = [r for r in rows if r[0]]
    answered = [r for r in ans if not r[1]]
    unans = [r for r in rows if not r[0]]
    return {"counts": cnt, "n": len(rows), "n_answerable": len(ans), "n_answered": len(answered),
            "halluc_answered": (float(np.mean([r[3] for r in answered])) if answered else math.nan),
            "correct_answered": (float(np.mean([r[2] for r in answered])) if answered else math.nan),
            "refusal_wrong_rate": (cnt["refused_wrongly"] / len(ans)) if ans else math.nan,
            "refusal_correct_rate": (cnt["refused_correctly"] / len(unans)) if unans else math.nan,
            "coverage": (len(answered) / len(ans)) if ans else math.nan}
