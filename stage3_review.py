"""Validate human review and invalidate only caches affected by approved edits."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prepare_review(raw, key, labels, edits):
    """Pure validation: no files are written until the complete review is valid."""
    if labels.get("sheet_id") != key["sheet_id"]:
        raise ValueError("Nhãn duyệt không khớp sheet_id của phiếu.")
    originals = {t["id"]: t for t in raw}
    if len(originals) != len(raw) or set(key["cases"].values()) != set(originals):
        raise ValueError("Phiếu và dữ liệu gốc không khớp hoặc trùng ID.")
    by_id = {}
    for row in labels["labels"]:
        code = row["code"]
        if code not in key["cases"]:
            raise ValueError(f"Mã câu lạ: {code}")
        qid = key["cases"][code]
        if qid in by_id or type(row.get("label")) is not int or row["label"] not in (0, 1, 2):
            raise ValueError(f"Nhãn thiếu, không hợp lệ hoặc lặp: {code}")
        by_id[qid] = row
    if set(by_id) != set(originals):
        raise ValueError("Còn câu chưa duyệt.")
    kept, changed_questions, changed_targets = [], set(), set()
    for qid, original in originals.items():
        row = by_id[qid]
        if row["label"] == 0:
            continue
        item = dict(original)
        if row["label"] == 2:
            edit = edits.get(qid, {})
            if not all(isinstance(edit.get(k), str) and edit[k].strip() for k in ("question", "ground_truth")):
                raise ValueError(f"{qid}: cần bản sửa question và ground_truth trong edits.json.")
            item["original"] = {k: original[k] for k in ("question", "ground_truth")}
            item.update({k: edit[k].strip() for k in ("question", "ground_truth")})
            item["edit_basis"] = edit.get("edit_basis", "human_review_note")
        if item["question"] != original["question"]:
            changed_questions.add(qid)
        if any(item[k] != original[k] for k in ("question", "ground_truth")):
            changed_targets.add(qid)
        item.update(reviewed=True, keep=True, review_note=row.get("note", ""),
                    review_sheet_id=key["sheet_id"])
        kept.append(item)
    return kept, changed_questions, changed_targets


def invalidate_caches(root, questions, targets, backup):
    """Back up before removing stale answers, scores and reference claims, including extra orders."""
    changed = []
    for fp in sorted(Path(root).rglob("*.jsonl")):
        if ".s3_backups" in fp.parts:
            continue
        if fp.name.startswith("answers"):
            ids = questions
        elif fp.name.startswith("scores") or fp.name == "gt_claims.jsonl":
            ids = targets
        else:
            continue
        if not ids:
            continue
        lines = fp.read_text(encoding="utf-8").splitlines(keepends=True)
        keep = [s for s in lines if not s.strip() or json.loads(s)["id"] not in ids]
        if len(keep) == len(lines):
            continue
        dest = backup / fp.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, dest)
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        tmp.replace(fp)
        changed.append({"file": str(fp.relative_to(root)), "removed": len(lines) - len(keep)})
    return changed


def apply_review_files(pairs, key_path, labels_path, edits_path, results_dir):
    raw = [t for src, _ in pairs if src.exists() for t in read_json(src)]
    key, labels = read_json(key_path), read_json(labels_path)
    edits = read_json(edits_path) if edits_path.exists() else {}
    kept, questions, targets = prepare_review(raw, key, labels, edits)
    # Compare with a previous import too, so a subsequent correction invalidates its caches.
    previous = {t["id"]: t for _, dst in pairs if dst.exists() for t in read_json(dst)}
    for t in kept:
        prev = previous.get(t["id"])
        if prev:
            questions.discard(t["id"])
            targets.discard(t["id"])
        if prev and prev["question"] != t["question"]:
            questions.add(t["id"])
        if prev and any(prev[k] != t[k] for k in ("question", "ground_truth")):
            targets.add(t["id"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = Path(results_dir) / ".s3_backups" / stamp
    changes = invalidate_caches(Path(results_dir), questions, targets, backup)
    counts = {}
    for src, dst in pairs:
        ids = {t["id"] for t in read_json(src)} if src.exists() else set()
        rows = [t for t in kept if t["id"] in ids]
        if not src.exists():
            continue
        if dst.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup / dst.name)
        dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[dst.name] = len(rows)
    audit = {"imported_at_utc": stamp, "labels_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
             "sheet_id": key["sheet_id"], "kept": counts, "cache_changes": changes,
             "question_edits": sorted(questions), "target_edits": sorted(targets)}
    (Path(key_path).parent / "import_info.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
