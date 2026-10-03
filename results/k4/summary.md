# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG — `k4`

LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · top-k=4 · version: xem `run_info.json`

## 1. Ba chỉ số chính

### Tất cả câu có đáp án (single-hop + multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     | 121 |         0.8586 | [0.821, 0.895]      |             0.743  | [0.695, 0.786]          |               0.4215 | [0.331, 0.512]            |
| langgraph_selfrag | 121 |         0.8916 | [0.862, 0.919]      |             0.6632 | [0.602, 0.721]          |               0.3388 | [0.256, 0.421]            |

### Câu hỏi một đoạn (single-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  86 |         0.8902 | [0.845, 0.933]      |             0.7591 | [0.706, 0.804]          |               0.3023 | [0.209, 0.395]            |
| langgraph_selfrag |  86 |         0.9053 | [0.872, 0.937]      |             0.7626 | [0.710, 0.805]          |               0.3256 | [0.233, 0.430]            |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  35 |         0.7756 | [0.724, 0.826]      |             0.7035 | [0.603, 0.788]          |               0.7143 | [0.571, 0.857]            |
| langgraph_selfrag |  35 |         0.8308 | [0.771, 0.891]      |             0.4191 | [0.283, 0.554]          |               0.3714 | [0.229, 0.543]            |

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
| langchain_rag     |              0.8586 |                    0.4215 |              0.1404 |                       0.4679 |               0.843  |         0.0992 |          0.8595 |      9.1155 |      1      |
| langgraph_selfrag |              0.8688 |                    0.3636 |              0.121  |                       0.4184 |               0.7686 |         0.1901 |          0.686  |     27.6821 |      8.1322 |

### Câu hỏi một đoạn (single-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8902 |                    0.3023 |              0.1017 |                       0.3291 |               0.8488 |         0.0814 |          0.9651 |      7.9267 |      1      |
| langgraph_selfrag |              0.8995 |                    0.3372 |              0.1028 |                       0.35   |               0.8837 |         0.0698 |          0.8953 |     26.3597 |      7.6047 |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.7756 |                    0.7143 |              0.2167 |                       0.8333 |               0.8286 |         0.1429 |          0.6    |     12.0363 |      1      |
| langgraph_selfrag |              0.7324 |                    0.4286 |              0.1802 |                       0.7222 |               0.4857 |         0.4857 |          0.1714 |     30.9314 |      9.4286 |

### Câu hỏi ngoài corpus (đúng = từ chối; answer_relevancy của câu từ chối = 0)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |                 nan |                         0 |                   0 |                          nan |                    1 |              1 |             nan |      8.2283 |      1      |
| langgraph_selfrag |                 nan |                         0 |                   0 |                          nan |                    1 |              1 |             nan |     27.8822 |     13.6957 |
