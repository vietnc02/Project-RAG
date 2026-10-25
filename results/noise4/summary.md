# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG — `noise4`

LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · top-k=4 · version: xem `run_info.json`

## 1. Ba chỉ số chính

### Tất cả câu có đáp án (single-hop + multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     | 121 |         0.8689 | [0.837, 0.900]      |             0.756  | [0.709, 0.797]          |               0.3967 | [0.306, 0.488]            |
| langgraph_selfrag | 121 |         0.8927 | [0.852, 0.928]      |             0.6405 | [0.579, 0.701]          |               0.2975 | [0.215, 0.372]            |

### Câu hỏi một đoạn (single-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  86 |         0.9249 | [0.895, 0.954]      |             0.7686 | [0.715, 0.814]          |               0.2558 | [0.163, 0.349]            |
| langgraph_selfrag |  86 |         0.9024 | [0.859, 0.940]      |             0.7514 | [0.697, 0.796]          |               0.314  | [0.221, 0.419]            |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  35 |         0.7261 | [0.661, 0.787]      |             0.7251 | [0.633, 0.801]          |               0.7429 | [0.600, 0.886]            |
| langgraph_selfrag |  35 |         0.8448 | [0.758, 0.925]      |             0.368  | [0.230, 0.502]          |               0.2571 | [0.114, 0.400]            |

*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*

- **faithfulness** (0–1, cao = tốt): câu trả lời được tách thành các claim nguyên tử; judge kiểm tra từng
  claim có suy ra được từ context không. Điểm = #claim được hỗ trợ / #claim. Câu từ chối không tính.
- **answer_relevancy** (0–1, cao = tốt): judge sinh ngược 3 câu hỏi từ câu trả lời; điểm = trung bình
  cosine (embedding) với câu hỏi gốc. Câu né tránh/từ chối = 0. Không dùng cho câu ngoài corpus.
- **hallucination_rate** (0–1, thấp = tốt): tỉ lệ câu hỏi có câu trả lời chứa ≥ 1 claim không được
  context hỗ trợ, tính trên toàn bộ câu hỏi.
- **Context chấm điểm**: full top-k ban đầu của câu hỏi gốc, giống nhau cho cả 2 hệ. Cột `*_used` chấm
  trên tài liệu thực sự đưa vào generate (Self-RAG: tài liệu đã lọc).
- **refusal_rate ở câu ngoài corpus** (cao = tốt): tỉ lệ hệ thống từ chối trả lời đúng (abstain).

## 2. Chỉ số phụ (chỉ để giải thích kết quả)

### Tất cả câu có đáp án (single-hop + multi-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8689 |                    0.3967 |              0.1464 |                       0.4364 |               0.8595 |         0.0909 |          0.8678 |     17.0512 |      1      |
| langgraph_selfrag |              0.8936 |                    0.3223 |              0.1089 |                       0.3789 |               0.7355 |         0.2149 |          0.6694 |     36.8613 |     14.0248 |

### Câu hỏi một đoạn (single-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.9249 |                    0.2558 |              0.0853 |                       0.2785 |               0.8721 |         0.0814 |          0.9651 |     16.7412 |      1      |
| langgraph_selfrag |              0.9083 |                    0.3256 |              0.0962 |                       0.3418 |               0.8721 |         0.0814 |          0.8721 |     35.3213 |     13.1279 |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.7261 |                    0.7429 |              0.2581 |                       0.8387 |               0.8286 |         0.1143 |          0.6286 |     17.8131 |      1      |
| langgraph_selfrag |              0.8213 |                    0.3143 |              0.1579 |                       0.5625 |               0.4    |         0.5429 |          0.1714 |     40.6454 |     16.2286 |

## Judge failures và truncate (log judge)

| system               | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   retries |   unresolved |   prompt_tokens_max |   prompt_tokens_median |
|:---------------------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|----------:|-------------:|--------------------:|-----------------------:|
| langchain_rag.p0     | correctness       |     110 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 350 |                    205 |
| langchain_rag.p0     | extract_claims    |     113 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                 264 |                    142 |
| langchain_rag.p0     | reverse_questions |     110 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 263 |                    134 |
| langchain_rag.p0     | verify_claims     |     110 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2503 |                   1791 |
| langchain_rag.p0     | (tổng)            |     443 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                2503 |                    169 |
| langgraph_selfrag.p0 | correctness       |      95 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 311 |                    196 |
| langgraph_selfrag.p0 | extract_claims    |      96 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                 200 |                    136 |
| langgraph_selfrag.p0 | reverse_questions |      95 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 191 |                    127 |
| langgraph_selfrag.p0 | verify_claims     |     190 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2534 |                   1265 |
| langgraph_selfrag.p0 | (tổng)            |     476 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                2534 |                    196 |

*judge_failures: số lần gọi judge không trả đủ verdict hợp lệ = json_error (JSON hỏng) + ollama_error (Ollama huỷ lần sinh) + lần gọi thiếu verdict; verdicts_missing: số claim / đoạn văn không có verdict (bị tính là "không"); output_cut: đầu ra chạm num_predict; truncated_prompt: prompt dài hơn num_ctx = 4096 bị Ollama cắt. Chỉ gồm các câu được chấm sau khi có log.*
