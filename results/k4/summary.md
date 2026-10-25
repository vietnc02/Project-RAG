# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG — `k4`

LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · top-k=4 · version: xem `run_info.json`

## 1. Ba chỉ số chính

### Tất cả câu có đáp án (single-hop + multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     | 121 |         0.8676 | [0.828, 0.902]      |             0.7404 | [0.693, 0.783]          |               0.405  | [0.314, 0.496]            |
| langgraph_selfrag | 121 |         0.8951 | [0.867, 0.922]      |             0.6483 | [0.586, 0.708]          |               0.3223 | [0.240, 0.405]            |

### Câu hỏi một đoạn (single-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  86 |         0.8933 | [0.845, 0.938]      |             0.756  | [0.703, 0.801]          |               0.2791 | [0.186, 0.372]            |
| langgraph_selfrag |  86 |         0.9107 | [0.879, 0.941]      |             0.7423 | [0.686, 0.790]          |               0.3023 | [0.209, 0.395]            |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  35 |         0.8002 | [0.754, 0.845]      |             0.702  | [0.602, 0.786]          |               0.7143 | [0.571, 0.857]            |
| langgraph_selfrag |  35 |         0.8275 | [0.780, 0.874]      |             0.4173 | [0.282, 0.551]          |               0.3714 | [0.229, 0.543]            |

### Câu hỏi ngoài corpus (đúng = từ chối; answer_relevancy của câu từ chối = 0)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  23 |            nan | [nan, nan]          |                  0 | [0.000, 0.000]          |                    0 | [0.000, 0.000]            |
| langgraph_selfrag |  23 |            nan | [nan, nan]          |                  0 | [0.000, 0.000]          |                    0 | [0.000, 0.000]            |

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
| langchain_rag     |              0.8676 |                    0.405  |              0.1284 |                       0.4495 |               0.8512 |         0.0992 |          0.8595 |     10.625  |      1      |
| langgraph_selfrag |              0.8748 |                    0.3719 |              0.1136 |                       0.4062 |               0.7438 |         0.2066 |          0.6777 |     26.3011 |      8.1818 |

### Câu hỏi một đoạn (single-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8933 |                    0.2791 |              0.0979 |                       0.3038 |               0.8605 |         0.0814 |          0.9651 |     10.2598 |      1      |
| langgraph_selfrag |              0.8981 |                    0.3488 |              0.0933 |                       0.3333 |               0.8488 |         0.093  |          0.8837 |     25.5603 |      7.6744 |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8002 |                    0.7143 |              0.1871 |                       0.8333 |               0.8286 |         0.1429 |          0.6    |     11.5226 |      1      |
| langgraph_selfrag |              0.7736 |                    0.4286 |              0.1786 |                       0.7222 |               0.4857 |         0.4857 |          0.1714 |     28.1211 |      9.4286 |

### Câu hỏi ngoài corpus (đúng = từ chối; answer_relevancy của câu từ chối = 0)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |                 nan |                         0 |                   0 |                          nan |                    1 |              1 |             nan |      8.2183 |      1      |
| langgraph_selfrag |                 nan |                         0 |                   0 |                          nan |                    1 |              1 |             nan |     25.1004 |     13.3043 |

## Judge failures và truncate (log judge)

| system               | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   retries |   unresolved |   prompt_tokens_max |   prompt_tokens_median |
|:---------------------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|----------:|-------------:|--------------------:|-----------------------:|
| langchain_rag.p0     | correctness       |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 384 |                    204 |
| langchain_rag.p0     | extract_claims    |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 244 |                    141 |
| langchain_rag.p0     | reverse_questions |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 231 |                    133 |
| langchain_rag.p0     | verify_claims     |     110 |                1 |            0 |              0 |            0 |                        1 |                  1 |                  0 |         1 |            0 |                1460 |                    983 |
| langchain_rag.p0     | (tổng)            |     437 |                1 |            0 |              0 |            0 |                        1 |                  1 |                  0 |         1 |            0 |                1460 |                    170 |
| langchain_rag.p1     | correctness       |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 384 |                    204 |
| langchain_rag.p1     | extract_claims    |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 244 |                    141 |
| langchain_rag.p1     | reverse_questions |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 231 |                    133 |
| langchain_rag.p1     | verify_claims     |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                1460 |                    983 |
| langchain_rag.p1     | (tổng)            |     436 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                1460 |                    169 |
| langchain_rag.p2     | correctness       |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 384 |                    204 |
| langchain_rag.p2     | extract_claims    |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 244 |                    141 |
| langchain_rag.p2     | reverse_questions |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 231 |                    133 |
| langchain_rag.p2     | verify_claims     |     109 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                1460 |                    983 |
| langchain_rag.p2     | (tổng)            |     436 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                1460 |                    169 |
| langgraph_selfrag.p0 | correctness       |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 471 |                    196 |
| langgraph_selfrag.p0 | extract_claims    |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 331 |                    136 |
| langgraph_selfrag.p0 | reverse_questions |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 318 |                    131 |
| langgraph_selfrag.p0 | verify_claims     |     179 |                1 |            0 |              1 |            0 |                        1 |                  9 |                  0 |         1 |            0 |                1457 |                    829 |
| langgraph_selfrag.p0 | (tổng)            |     467 |                1 |            0 |              1 |            0 |                        1 |                  9 |                  0 |         1 |            0 |                1457 |                    190 |
| langgraph_selfrag.p1 | correctness       |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 471 |                    196 |
| langgraph_selfrag.p1 | extract_claims    |      97 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                 331 |                    136 |
| langgraph_selfrag.p1 | reverse_questions |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 318 |                    131 |
| langgraph_selfrag.p1 | verify_claims     |     178 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                1456 |                    829 |
| langgraph_selfrag.p1 | (tổng)            |     467 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                1456 |                    190 |
| langgraph_selfrag.p2 | correctness       |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 471 |                    196 |
| langgraph_selfrag.p2 | extract_claims    |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 331 |                    136 |
| langgraph_selfrag.p2 | reverse_questions |      96 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 318 |                    131 |
| langgraph_selfrag.p2 | verify_claims     |     179 |                1 |            0 |              1 |            0 |                        1 |                  9 |                  0 |         1 |            0 |                1456 |                    829 |
| langgraph_selfrag.p2 | (tổng)            |     467 |                1 |            0 |              1 |            0 |                        1 |                  9 |                  0 |         1 |            0 |                1456 |                    190 |

*judge_failures: số lần gọi judge không trả đủ verdict hợp lệ = json_error (JSON hỏng) + ollama_error (Ollama huỷ lần sinh) + lần gọi thiếu verdict; verdicts_missing: số claim / đoạn văn không có verdict (bị tính là "không"); output_cut: đầu ra chạm num_predict; truncated_prompt: prompt dài hơn num_ctx = 4096 bị Ollama cắt. Chỉ gồm các câu được chấm sau khi có log.*
