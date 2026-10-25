"""Kết quả duyệt tay câu single-hop cho các bài bổ sung của ý 2 (đối chiếu từng câu với đoạn gốc trong
singlehop_kg_raw.json).

    python testset/review_singlehop_kg.py      -> testset/singlehop_kg.json

Tiêu chí giống nội dung 1: câu hỏi tự đứng được (không "the passage / the paper / the article"), không hỏi công
thức bị lỗi ký tự; đáp án chuẩn chỉ chứa thông tin có trong đoạn gốc.
- DROP: câu bị loại kèm lý do
- EDITS: id -> (question mới, ground_truth mới)
"""
import json
from pathlib import Path

DROP = {
    "s003": "hỏi ký hiệu của một công thức (QFT của tín hiệu tuần hoàn, phương trình 34), đáp án là công thức",
}

EDITS = {
    "s001": ("In variational quantum circuits, what name is given in quantum simulation to the approach that uses "
             "parametrized random circuits of varying depth?",
             "It is called a \"hardware efficient ansatz\": instead of a parameterization inspired by physical domain "
             "knowledge, the circuit is a parametrized random circuit of varying depth."),
    "s002": ("Which states are used in the Vacuum+Weak decoy-state method for quantum key distribution, and what do "
             "Alice and Bob gain by measuring the decoy states?",
             "Only a few states are used: the vacuum, a weak decoy state with mean photon number µ_decoy ≪ 1 and a "
             "signal state with µ = O(1). By measuring the yields and QBER of the decoy states, Alice and Bob obtain "
             "reliable bounds on the single-photon parameters (Ω and e1), surpassing prior results."),
    "s004": ("In Shor's quantum algorithm for discrete logarithms, with what probability is each good (c, d) pair "
             "generated, and what fraction of the possible c values belong to a good pair?",
             "Each good (c, d) pair is generated with probability at least 1/(20q²), and at least a twelfth of the "
             "possible c's are in a good (c, d) pair."),
    "s005": ("Why is it difficult to foresee the future applications of quantum computing, and what recent industry "
             "trend surprised many quantum researchers in academia?",
             "Because quantum computing technology is so different from today's information technology, there is only "
             "a very limited ability to glimpse its future applications or project when they will come to fruition. "
             "A recent surge of investment by large public companies and startups happened sooner and more suddenly "
             "than most researchers expected."),
    "s006": ("In quantum public key distribution with polarized photons (BB84), how do Alice and Bob test for "
             "eavesdropping, and how are the remaining bits used?",
             "They publicly compare a random subset (say one third) of the correctly received bits, sacrificing their "
             "secrecy, so that eavesdropping on more than a few photons is unlikely to escape detection. If all "
             "comparisons agree, the remaining bits sent and received in the same basis can be used as a one-time pad; "
             "when the pad is used up the protocol is repeated."),
    "s007": ("How do local parameter-update optimizers for variational quantum algorithms work, and what has been "
             "proposed to speed up their convergence?",
             "They perform local parameter updates sequentially over all parameters (or subsets) and iterate, giving "
             "a gradient-free method that does not depend on hyper-parameters. A variant using Anderson acceleration, "
             "which adds a linear combination of prior steps to each new update step, has been proposed to speed up "
             "convergence."),
}


def main():
    here = Path(__file__).resolve().parent
    items = json.loads((here / "singlehop_kg_raw.json").read_text(encoding="utf-8"))
    assert set(DROP) | set(EDITS) == {t["id"] for t in items} and not set(DROP) & set(EDITS)
    out = []
    for t in items:
        if t["id"] in DROP:
            continue
        t["original"] = {"question": t["question"], "ground_truth": t["ground_truth"]}
        t["question"], t["ground_truth"] = EDITS[t["id"]]
        t["type"], t["reviewed"], t["keep"] = "single", True, True
        out.append(t)
    (here / "singlehop_kg.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"giữ {len(out)} / {len(items)} câu -> singlehop_kg.json (loại {len(DROP)})")


if __name__ == "__main__":
    main()
