# Ý 1 — LangChain RAG thuần vs LangGraph Self-RAG (server local)

## Chạy nhanh (sau khi clone repo)
Repo đã có sẵn dữ liệu (chunks + Chroma), không cần build lại.
1. Cài [Ollama](https://ollama.com) 0.35.x và tải 3 model:
   ```powershell
   ollama pull qwen2.5:7b; ollama pull llama3.1:8b; ollama pull nomic-embed-text
   ```
2. Cài thư viện (Python 3.14): `pip install -r requirements.txt`
3. Chạy:
   ```powershell
   python rag_langchain.py "What is a network tarpit?"   # hệ A
   python rag_selfrag.py   "What is a network tarpit?"   # hệ B (in ra trace)
   python evaluate.py                                    # chạy lại toàn bộ đánh giá
   ```
   Lần chạy đầu, `data/chroma.zip` tự giải nén ra `data/chroma/` (có kiểm tra sha256).

## Môi trường
| Thành phần | Version |
|---|---|
| Python | 3.14.6 |
| langchain / langchain-core | 1.4.3 / 1.6.6 |
| langgraph | 1.2.12 |
| langchain-ollama / langchain-chroma | 1.1.0 / 1.1.0 |
| chromadb | 1.5.9 |
| Ollama server | 0.35.0; phần cuối run `k8` chạy trên 0.35.1 (Ollama tự cập nhật, digest model không đổi). `http://localhost:11434` |
| LLM sinh câu trả lời | `qwen2.5:7b` Q4_K_M (digest `845dbda0ea48`) |
| Judge | `llama3.1:8b` Q4_K_M (digest `46e0c10c039e`) |
| Embedding | `nomic-embed-text` F16 (digest `0a109f422b47`) |
| Phần cứng | i3-12100F, RAM 16 GB, GTX 1660 Ti 6 GB |

Toàn bộ thư viện Python đã pin trong `requirements.txt`. Mỗi lần chạy `evaluate.py` cũng ghi lại version vào thư mục kết quả.
Trên máy tác giả: venv `D:\RAG_env\venv` (`D:\RAG_env\venv\Scripts\Activate.ps1`), model Ollama ở `D:\ollama_models`.

## Dữ liệu (chunks + Chroma, có trong repo)
| File | Nội dung |
|---|---|
| `data/chunks.jsonl` | 6.649 chunk (văn bản + metadata nguồn/trang) từ 95 bài báo, chunk 1000 / overlap 150 |
| `data/chroma.zip` | Vector DB Chroma đúng như lúc chạy thí nghiệm (collection `papers`, cosine, 6.649 vector) |
| `data/MANIFEST.json` | Số lượng, sha256 của 2 file trên, tham số chunking, version thư viện / Ollama / model |

`data/chroma/` (bản giải nén) không commit vì `chroma.sqlite3` bị ghi lại mỗi lần mở.
```powershell
python ingest.py --restore       # giải nén lại chroma.zip (thường tự động)
python ingest.py --from-chunks   # embed lại từ data/chunks.jsonl
python ingest.py --reset         # build lại từ đầu từ PDF trong papers/
python ingest.py --snapshot      # (sau khi build lại) cập nhật chroma.zip + MANIFEST.json
```

## Quy trình
```powershell
python ingest.py --reset                 # 1. (tuỳ chọn) build lại Chroma từ PDF
python rag_langchain.py "câu hỏi"        #    thử hệ thống A
python rag_selfrag.py   "câu hỏi"        #    thử hệ thống B (in ra trace)
python generate_testset.py               # 2a. sinh câu single-hop -> testset/testset.json
python generate_testset.py --multihop 60 # 2b. sinh câu multi-hop  -> testset/multihop_raw.json
python generate_testset.py --multihop 40 --append --seed 7   #  sinh thêm, không lặp cặp bài đã có
#   -> DUYỆT TAY từng câu (đối chiếu đoạn gốc), kết quả duyệt ghi trong:
python testset/review_edits.py           #    single-hop -> testset/testset.json
python testset/review_multihop.py        #    multi-hop  -> testset/multihop.json
#   -> câu ngoài corpus viết tay: testset/unanswerable.json (có kiểm tra không xuất hiện trong chunks)
python evaluate.py --run k4                                                    # 3a. top-k = 4
python evaluate.py --run k8 --top-k 8 --testset testset/testset.json testset/multihop.json  # 3b. retrieval nhiễu
```
Judge mặc định là `llama3.1:8b` (khác họ với LLM sinh câu trả lời để tránh tự chấm bài mình).
Đổi judge: `$env:JUDGE_MODEL='<model>'; python evaluate.py --run <tên mới>`.

## Hai hệ thống (dùng CHUNG retriever, top-k, prompt sinh câu trả lời, LLM)
- **A. LangChain RAG** (`rag_langchain.py`): retrieve top-k → generate. 1 lần gọi LLM.
- **B. LangGraph Self-RAG** (`rag_selfrag.py`): retrieve → chấm độ liên quan từng tài liệu →
  generate → tự kiểm tra *grounded* (hallucination, prompt có few-shot) → tự kiểm tra *useful* →
  nếu sai thì sinh lại (prompt chặt hơn) / viết lại truy vấn / từ chối trả lời.

## Bộ câu hỏi
| File | Loại | Số câu | Mục đích |
|---|---|---|---|
| `testset/testset.json` | single-hop: hỏi sự kiện trong 1 đoạn | 86 (sinh 93, duyệt tay) | trường hợp cơ bản |
| `testset/multihop.json` | multi-hop: cần tổng hợp 2 bài báo | 35 (sinh 71, duyệt tay) | câu hỏi khó, cần truy xuất nhiều nguồn |
| `testset/unanswerable.json` | ngoài corpus | 23 (viết tay, đã kiểm tra không có trong chunks) | hệ thống phải biết từ chối (abstain) |
| (`--top-k 8`) | retrieval nhiễu | single + multi-hop | context chứa nhiều đoạn không liên quan |

## Ba chỉ số so sánh (`evaluate.py`, định nghĩa theo RAGAS)
| Chỉ số | Định nghĩa |
|---|---|
| faithfulness ↑ | #claim được context hỗ trợ / #claim |
| answer_relevancy ↑ | cosine(câu hỏi, 3 câu hỏi sinh ngược từ câu trả lời); 0 nếu né tránh / từ chối |
| hallucination_rate ↓ | % câu hỏi có câu trả lời chứa ≥ 1 claim không được context hỗ trợ |

Faithfulness / hallucination của **cả 2 hệ** được chấm trên cùng một context: full top-k ban đầu của câu hỏi
gốc (Self-RAG lọc bớt tài liệu trước khi sinh, nếu chấm trên tài liệu đã lọc thì không công bằng).
Chỉ số phụ chỉ để giải thích: `*_used` (chấm trên tài liệu đưa vào generate), refusal_rate, answer_correctness,
retrieval_hit, claim_halluc_rate, latency_s, llm_calls.

Kết quả: `results/<run>/summary.md`, `summary.csv` (kèm CI 95% bootstrap), `per_question.csv`, `run_info.json`.
