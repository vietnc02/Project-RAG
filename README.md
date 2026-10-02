# Ý 1 — LangChain RAG thuần vs LangGraph Self-RAG (server local)

## Môi trường (đã cài sẵn)
| Thành phần | Vị trí |
|---|---|
| Python venv | `D:\RAG_env\venv` |
| Ollama | `D:\Ollama` (model lưu ở `D:\ollama_models`, biến `OLLAMA_MODELS`) |
| Vector DB (Chroma) + chunks | `D:\RAG_env\data` |
| LLM / Judge / Embedding | `qwen2.5:7b` / `llama3.1:8b` / `nomic-embed-text` (Ollama, `http://localhost:11434`) |

Kích hoạt venv (PowerShell): `D:\RAG_env\venv\Scripts\Activate.ps1`

## Quy trình
```powershell
python ingest.py                       # 1. PDF -> chunk -> Chroma (1 lần)
python rag_langchain.py "câu hỏi"      #    thử hệ thống A
python rag_selfrag.py   "câu hỏi"      #    thử hệ thống B (in ra trace)
python generate_testset.py             # 2. sinh testset/testset.json
#   -> DUYỆT TAY: sửa câu hỏi/đáp án, đặt "reviewed": true, câu kém đặt "keep": false
python evaluate.py                     # 3. chạy + chấm điểm -> results/
```
Judge mặc định là `llama3.1:8b` (khác họ với LLM sinh câu trả lời để tránh tự chấm bài mình).
Đổi judge: `$env:JUDGE_MODEL='<model>'; python evaluate.py`
(xoá `results/scores_*.jsonl` trước khi chấm lại; `answers_*.jsonl` giữ nguyên).

## Hai hệ thống (dùng CHUNG retriever, top-k, prompt sinh câu trả lời, LLM)
- **A. LangChain RAG** (`rag_langchain.py`): retrieve top-k → generate. 1 lần gọi LLM.
- **B. LangGraph Self-RAG** (`rag_selfrag.py`): retrieve → chấm độ liên quan từng tài liệu →
  generate → tự kiểm tra *grounded* (hallucination) → tự kiểm tra *useful* →
  nếu sai thì sinh lại (prompt chặt hơn) / viết lại truy vấn / từ chối trả lời.

## Chỉ số (`evaluate.py`)
| Chỉ số | Định nghĩa |
|---|---|
| faithfulness | #claim được context hỗ trợ / #claim (RAGAS) |
| answer_relevancy | cosine(câu hỏi, 3 câu hỏi sinh ngược từ câu trả lời); 0 nếu né tránh (RAGAS) |
| hallucination_rate | % câu trả lời có ≥1 claim không được hỗ trợ (trên tổng số câu hỏi) |
| claim_halluc_rate | % claim không được hỗ trợ |
| answer_correctness, refusal_rate, retrieval_hit, latency_s, llm_calls | chỉ số phụ |

Kết quả: `results/summary.md`, `summary.csv` (kèm CI 95% bootstrap), `per_question.csv`.
