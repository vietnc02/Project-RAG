import json
import tempfile
import unittest
from pathlib import Path

from stage3_review import apply_review_files, invalidate_caches, prepare_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.raw = [{"id": "t1", "question": "Q?", "ground_truth": "A", "reviewed": False}]
        self.key = {"sheet_id": "abc", "cases": {"Q001": "t1"}}
        self.labels = {"sheet_id": "abc", "labels": [{"code": "Q001", "label": 2, "note": "Correct answer"}]}

    def test_reference_only_edit_keeps_answers_but_invalidates_scores(self):
        kept, questions, targets = prepare_review(self.raw, self.key, self.labels,
                                                  {"t1": {"question": "Q?", "ground_truth": "B"}})
        self.assertEqual(questions, set())
        self.assertEqual(targets, {"t1"})
        self.assertTrue(kept[0]["reviewed"])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("answers.jsonl", "scores.p0.jsonl", "scores.jsonl", "gt_claims.jsonl"):
                fp = root / "compression" / name
                fp.parent.mkdir(exist_ok=True)
                fp.write_text('{"id":"t1"}\n{"id":"unchanged"}\n', encoding="utf-8")
            changed = invalidate_caches(root, questions, targets, root / ".s3_backups" / "test")
            self.assertEqual(len(changed), 3)
            self.assertIn('"t1"', (root / "compression/answers.jsonl").read_text())
            self.assertNotIn('"t1"', (root / "compression/gt_claims.jsonl").read_text())
            self.assertIn('"t1"', (root / ".s3_backups/test/compression/gt_claims.jsonl").read_text())

    def test_invalid_or_duplicate_labels_fail_before_writing(self):
        for rows in ([], [self.labels["labels"][0]] * 2, [{"code": "Q001", "label": 3}]):
            with self.assertRaises(ValueError):
                prepare_review(self.raw, self.key, {"sheet_id": "abc", "labels": rows}, {})
        with self.assertRaises(ValueError):
            prepare_review(self.raw, self.key, {**self.labels, "sheet_id": "wrong"}, {})

    def test_reimport_does_not_delete_regenerated_answers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payloads = {"raw.json": self.raw, "key.json": self.key, "labels.json": self.labels,
                        "edits.json": {"t1": {"question": "Revised?", "ground_truth": "B"}}}
            for name, value in payloads.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            results = root / "results"
            results.mkdir()
            args = ([(root / "raw.json", root / "final.json")], root / "key.json", root / "labels.json",
                    root / "edits.json", results)
            apply_review_files(*args)
            (results / "answers.jsonl").write_text('{"id":"t1","answer":"new"}\n', encoding="utf-8")
            audit = apply_review_files(*args)
            self.assertEqual(audit["cache_changes"], [])
            self.assertIn('"new"', (results / "answers.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
