# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG — `k8`

LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · top-k=8 · version: xem `run_info.json`

## 1. Ba chỉ số chính

### Tất cả câu có đáp án (single-hop + multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     | 121 |         0.8797 | [0.846, 0.911]      |             0.7685 | [0.727, 0.804]          |               0.405  | [0.314, 0.488]            |
| langgraph_selfrag | 121 |         0.9078 | [0.880, 0.934]      |             0.6797 | [0.621, 0.737]          |               0.2975 | [0.215, 0.380]            |

### Câu hỏi một đoạn (single-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  86 |         0.8977 | [0.862, 0.932]      |             0.777  | [0.730, 0.815]          |               0.3372 | [0.244, 0.442]            |
| langgraph_selfrag |  86 |         0.9197 | [0.889, 0.949]      |             0.7759 | [0.728, 0.815]          |               0.2791 | [0.186, 0.384]            |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  35 |         0.8341 | [0.772, 0.893]      |             0.7477 | [0.665, 0.818]          |               0.5714 | [0.400, 0.743]            |
| langgraph_selfrag |  35 |         0.8569 | [0.796, 0.913]      |             0.4433 | [0.304, 0.578]          |               0.3429 | [0.200, 0.486]            |

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
| langchain_rag     |              0.8797 |                    0.405  |              0.131  |                       0.4336 |               0.9091 |         0.0661 |          0.9174 |     16.9517 |      1      |
| langgraph_selfrag |              0.8745 |                    0.3967 |              0.0948 |                       0.36   |               0.7851 |         0.1736 |          0.7438 |     36.2404 |     12.9008 |

### Câu hỏi một đoạn (single-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8977 |                    0.3372 |              0.1198 |                       0.358  |               0.907  |         0.0581 |          0.9767 |     16.6713 |      1      |
| langgraph_selfrag |              0.9022 |                    0.3488 |              0.0838 |                       0.2963 |               0.8953 |         0.0581 |          0.9419 |     35.6851 |     11.7791 |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8341 |                    0.5714 |              0.1531 |                       0.625  |               0.9143 |         0.0857 |          0.7714 |     17.6406 |      1      |
| langgraph_selfrag |              0.7564 |                    0.5143 |              0.1304 |                       0.6316 |               0.5143 |         0.4571 |          0.2571 |     37.6049 |     15.6571 |

## Judge failures và truncate (log judge)

| system               | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   retries |   unresolved |   prompt_tokens_max |   prompt_tokens_median |
|:---------------------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|----------:|-------------:|--------------------:|-----------------------:|
| langchain_rag.p0     | correctness       |     113 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 385 |                    204 |
| langchain_rag.p0     | extract_claims    |     115 |                2 |            2 |              0 |            2 |                        0 |                  0 |                  0 |         2 |            0 |                 286 |                    141 |
| langchain_rag.p0     | reverse_questions |     113 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 285 |                    135 |
| langchain_rag.p0     | verify_claims     |     113 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2789 |                   1769 |
| langchain_rag.p0     | (tổng)            |     454 |                2 |            2 |              0 |            2 |                        0 |                  0 |                  0 |         2 |            0 |                2789 |                    171 |
| langgraph_selfrag.p0 | correctness       |     100 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 413 |                    196 |
| langgraph_selfrag.p0 | extract_claims    |     100 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 273 |                    135 |
| langgraph_selfrag.p0 | reverse_questions |     100 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 260 |                    128 |
| langgraph_selfrag.p0 | verify_claims     |     197 |                1 |            0 |              1 |            0 |                        1 |                  5 |                  0 |         1 |            0 |                2794 |                   1468 |
| langgraph_selfrag.p0 | (tổng)            |     497 |                1 |            0 |              1 |            0 |                        1 |                  5 |                  0 |         1 |            0 |                2794 |                    197 |

*judge_failures: số lần gọi judge không trả đủ verdict hợp lệ = json_error (JSON hỏng) + ollama_error (Ollama huỷ lần sinh) + lần gọi thiếu verdict; verdicts_missing: số claim / đoạn văn không có verdict (bị tính là "không"); output_cut: đầu ra chạm num_predict; truncated_prompt: prompt dài hơn num_ctx = 4096 bị Ollama cắt. Chỉ gồm các câu được chấm sau khi có log.*
