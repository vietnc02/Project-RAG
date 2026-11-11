import json
import tempfile
import unittest
from pathlib import Path

from stage3_runtime import record_attempt


class RuntimeTests(unittest.TestCase):
    def info(self, model="qwen2.5:7b"):
        return {"top_k": 4, "num_ctx": 4096, "testset": {"s3": "hash"},
                "environment": {"models": {"llm": {"name": model, "digest": "123"}}}}

    def test_resume_records_current_inputs_without_losing_original(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run_info.json"
            record_attempt(p, self.info(), answers_only=True)
            updated = {**self.info(), "testset": {"s3": "newhash"}}
            record_attempt(p, updated)
            saved = json.loads(p.read_text())
            self.assertEqual(saved["testset"]["s3"], "hash")
            self.assertEqual(saved["attempts"][-1]["testset"]["s3"], "newhash")
            self.assertTrue(saved["attempts"][0]["answers_only"])

    def test_different_generator_cannot_reuse_old_run(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run_info.json"
            record_attempt(p, self.info())
            with self.assertRaises(ValueError):
                record_attempt(p, self.info("llama3.1:8b"))


if __name__ == "__main__":
    unittest.main()
