# Kết quả ý 1: LangChain RAG vs LangGraph Self-RAG

LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · top-k=4

## 1. Ba chỉ số chính

| system            |   n |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |
|:------------------|----:|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|
| langchain_rag     |  86 |         0.8945 | [0.844, 0.939]      |             0.7577 | [0.704, 0.802]          |               0.2791 | [0.186, 0.372]            |
| langgraph_selfrag |  86 |         0.9231 | [0.891, 0.953]      |             0.7028 | [0.636, 0.762]          |               0.2558 | [0.163, 0.349]            |

*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*

- **faithfulness** (0–1, cao = tốt): câu trả lời được tách thành các claim nguyên tử; judge kiểm tra
  từng claim có suy ra được từ context đã truy xuất không. Điểm = #claim được hỗ trợ / #claim.
  Câu từ chối trả lời không có claim nên không tính vào trung bình.
- **answer_relevancy** (0–1, cao = tốt): judge sinh ngược 3 câu hỏi từ câu trả lời; điểm = trung bình
  cosine similarity (embedding) giữa câu hỏi gốc và 3 câu hỏi đó. Câu trả lời né tránh/từ chối = 0.
- **hallucination_rate** (0–1, thấp = tốt): tỉ lệ câu hỏi mà câu trả lời có ít nhất 1 claim
  không được context hỗ trợ, tính trên toàn bộ câu hỏi.

## 2. Chỉ số phụ (giải thích kết quả)

| system            |   claim_halluc_rate |   halluc_rate_among_answered |   answer_correctness |   refusal_rate |   retrieval_hit |   latency_s |   llm_calls |
|:------------------|--------------------:|-----------------------------:|---------------------:|---------------:|----------------:|------------:|------------:|
| langchain_rag     |              0.0978 |                       0.3038 |               0.8605 |         0.0814 |          0.9651 |     10.8164 |      1      |
| langgraph_selfrag |              0.0765 |                       0.2973 |               0.8256 |         0.1395 |          0.8837 |     29.6978 |      8.1047 |

- **claim_halluc_rate**: tỉ lệ claim sai trên tổng số claim. Cho biết mức độ hallucination
  (sai 1 chi tiết nhỏ hay sai toàn bộ), bổ sung cho hallucination_rate vốn chỉ đếm 0/1 theo câu.
- **halluc_rate_among_answered**: hallucination_rate chỉ tính trên các câu có trả lời (bỏ câu từ chối).
  Dùng để kiểm tra hệ thống có giảm hallucination thật hay chỉ nhờ từ chối nhiều hơn.
- **answer_correctness**: judge so câu trả lời với đáp án chuẩn (0/1). Đảm bảo việc giảm hallucination
  không làm giảm độ đúng của câu trả lời.
- **refusal_rate**: tỉ lệ câu hệ thống trả lời "không tìm thấy trong tài liệu". Self-RAG có cơ chế
  abstain nên tỉ lệ này thường cao hơn; cần đọc cùng answer_relevancy (câu từ chối = 0 điểm).
- **retrieval_hit**: tỉ lệ câu hỏi mà bài báo nguồn của câu hỏi có trong context. Cho biết lỗi đến từ
  khâu truy xuất hay khâu sinh câu trả lời.
- **latency_s / llm_calls**: thời gian và số lần gọi LLM trung bình mỗi câu, tức chi phí của
  vòng tự kiểm tra trong Self-RAG.
