"""Materialize the explicit corrections in the first human review, retaining its provenance.

Run from the repository root with --labels PATH; inspect edits.json before --apply.
This is a record of this review round, not an automatic human annotator.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args()
    key = json.loads((DIR / "key.json").read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8-sig"))
    if labels["sheet_id"] != key["sheet_id"]:
        raise SystemExit("Nhãn không khớp phiếu.")
    raw = {t["id"]: t for name in ("singlehop_s3_raw", "multihop_s3_raw")
           for t in json.loads((ROOT / "testset" / f"{name}.json").read_text(encoding="utf-8"))}
    edits = {}
    for row in labels["labels"]:
        if row["label"] != 2:
            continue
        qid, note = key["cases"][row["code"]], row["note"]
        quotes = re.findall(r'"([^"]+)"', note)
        questions = [s for s in quotes if s.endswith("?")]
        answers = [s for s in quotes if not s.endswith("?")]
        if len(questions) > 1 or len(answers) > 1 and qid != "t098":
            raise ValueError(f"Cần đối chiếu thủ công: {qid}")
        edit = {"question": questions[0] if questions else raw[qid]["question"],
                "ground_truth": answers[0] if answers else raw[qid]["ground_truth"],
                "edit_basis": "explicit_human_review_note", "note": note}
        # Edits expressed as Vietnamese instructions rather than a complete replacement question.
        if qid == "t093":
            edit["question"] = "What is the practical decoy-state QKD key-generation rate in terms of Q_μ, Q_1, f(E_μ), H_2(E_μ), and H_2(e_1), and what does the factor q represent?"
        elif qid == "t169":
            edit["question"] = "In the localized-cost QCNN of Beyond Barren Plateaus: A Scalable Quantum Convolutional Architecture for High-Fidelity Image Classification, what is the pooling unitary V_pool(θ), with j the discarded qubit and k the surviving qubit?"
        elif qid == "h008":
            edit["question"] = raw[qid]["question"].replace("hardware with non-nearest-neighbor connectivity", "hardware limited to nearest-neighbor connectivity")
        elif qid == "h017":
            edit["question"] = "How do the computing resources and runtime reported for the classical cryptanalytic attack on SIKE compare with the computing resources needed to threaten RSA/ECC using Shor’s algorithm?"
        elif qid == "t206":
            edit["ground_truth"] += " This aggregates weighted risk across the organization’s n data types."
        # Remove references to unnamed passages from retained questions, preserving their meaning.
        if qid in ("h015", "h035"):
            edit["question"] = re.sub(r", according to (?:the discussions in both papers|the passages)\?", "?", edit["question"])
        edits[qid] = edit
    (DIR / "edits.json").write_text(json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(edits)} corrections -> {DIR / 'edits.json'}")


if __name__ == "__main__":
    main()
