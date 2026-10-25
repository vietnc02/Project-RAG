# Baseline không truy xuất (closed-book)

LLM `qwen2.5:7b` không có context; judge `llama3.1:8b` (prompt CORRECTNESS của ý 1–2).

| loại câu | n | answer correctness | refusal |
|---|---|---|---|
| single | 92 | 0.315 | 0.554 |
| multihop | 65 | 0.538 | 0.446 |
| unanswerable | 23 | 0.609 | 0.609 |
