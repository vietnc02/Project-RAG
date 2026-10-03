# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG — `k8`

LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · top-k=8 · version: xem `run_info.json`

## 1. Ba chỉ số chính

### Tất cả câu có đáp án (single-hop + multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     | 121 |         0.8785 | [0.845, 0.910]      |             0.7674 | [0.727, 0.803]          |                0.405 | [0.314, 0.488]            |
| langgraph_selfrag | 121 |         0.8996 | [0.869, 0.928]      |             0.6732 | [0.614, 0.731]          |                0.314 | [0.231, 0.397]            |

### Câu hỏi một đoạn (single-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  86 |         0.8977 | [0.862, 0.932]      |             0.7757 | [0.729, 0.814]          |               0.3372 | [0.244, 0.442]            |
| langgraph_selfrag |  86 |         0.9203 | [0.889, 0.950]      |             0.7763 | [0.728, 0.816]          |               0.2791 | [0.186, 0.384]            |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  35 |         0.8299 | [0.768, 0.892]      |             0.747  | [0.664, 0.816]          |               0.5714 | [0.400, 0.743]            |
| langgraph_selfrag |  35 |         0.8065 | [0.727, 0.877]      |             0.4199 | [0.280, 0.553]          |               0.4    | [0.229, 0.571]            |

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
| langchain_rag     |              0.8785 |                    0.405  |              0.1334 |                       0.4336 |               0.9091 |         0.0661 |          0.9174 |     17.5337 |      1      |
| langgraph_selfrag |              0.8759 |                    0.3719 |              0.1076 |                       0.3838 |               0.7851 |         0.1818 |          0.7438 |     38.2326 |     12.9091 |

### Câu hỏi một đoạn (single-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8977 |                    0.3372 |              0.1201 |                       0.358  |               0.907  |         0.0581 |          0.9767 |     17.2435 |      1      |
| langgraph_selfrag |              0.89   |                    0.3488 |              0.0836 |                       0.2963 |               0.8953 |         0.0581 |          0.9419 |     38.0117 |     11.7791 |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system            |   faithfulness_used |   hallucination_rate_used |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|--------------------------:|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.8299 |                    0.5714 |              0.1598 |                       0.625  |               0.9143 |         0.0857 |          0.7714 |     18.2469 |      1      |
| langgraph_selfrag |              0.8124 |                    0.4286 |              0.1826 |                       0.7778 |               0.5143 |         0.4857 |          0.2571 |     38.7751 |     15.6857 |
