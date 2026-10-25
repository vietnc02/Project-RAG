# Human judge so với LLM-judge (ý 2)

36 câu trả lời (18 câu hỏi × 2 hệ; single-hop 9, multi-hop 9 câu hỏi), judge gốc `llama3.1:8b` (results/kg30).

## 1. Mức agreement

| nhãn                                   |   n_labels |   n_answers |   human_pos |   judge_pos |   agreement | agreement_ci95   |   kappa | kappa_ci95      | mức (Landis & Koch)   |
|:---------------------------------------|-----------:|------------:|------------:|------------:|------------:|:-----------------|--------:|:----------------|:----------------------|
| Passage hữu ích (retrieval precision)  |        139 |          35 |       0.209 |       0.532 |       0.547 | [0.453, 0.64]    |   0.126 | [-0.008, 0.274] | rất thấp              |
| Câu trả lời đúng (answer correctness)  |         36 |          36 |       0.417 |       0.833 |       0.528 | [0.361, 0.694]   |   0.15  | [-0.056, 0.369] | rất thấp              |
| Có ý không được hỗ trợ (hallucination) |         32 |          32 |       0.156 |       0.531 |       0.625 | [0.469, 0.781]   |   0.281 | [0.077, 0.548]  | thấp                  |

*CI 95%: bootstrap theo câu trả lời. κ theo Landis & Koch: < 0,2 rất thấp, 0,2–0,4 thấp, 0,4–0,6 trung bình, 0,6–0,8 khá, > 0,8 gần như hoàn toàn. Hallucination chỉ tính câu không refuse; judge gốc không lưu verdict từng ý nên so ở mức câu trả lời. Khi một nhãn chiếm phần lớn (human_pos / judge_pos gần 0 hoặc 1, vd. answer correctness), κ có thể thấp dù % agreement cao (prevalence paradox), nên đọc κ cùng % agreement.*

Faithfulness (mức câu trả lời, n = 32): trung bình human judge 0.962, judge 0.869; sai khác tuyệt đối trung bình 0.102; tương quan Pearson 0.431.

## 2. GraphRAG − Vector RAG trên cùng mẫu: nhãn human judge và nhãn judge

| metric            | who   |   n_pairs |   graphrag |   vector_rag |   diff | ci95            | significant   |
|:------------------|:------|----------:|-----------:|-------------:|-------:|:----------------|:--------------|
| context_precision | human |        18 |      0.153 |        0.25  | -0.097 | [-0.208, 0.014] | False         |
| context_precision | judge |        17 |      0.515 |        0.549 | -0.034 | [-0.176, 0.098] | False         |
| faithfulness      | human |        15 |      0.99  |        0.928 |  0.062 | [0.01, 0.138]   | True          |
| faithfulness      | judge |        15 |      0.844 |        0.893 | -0.049 | [-0.145, 0.056] | False         |
| hallucination     | human |        18 |      0.056 |        0.222 | -0.167 | [-0.333, 0.0]   | False         |
| hallucination     | judge |        18 |      0.556 |        0.389 |  0.167 | [-0.111, 0.444] | False         |
| correct           | human |        18 |      0.333 |        0.5   | -0.167 | [-0.389, 0.056] | False         |
| correct           | judge |        18 |      0.833 |        0.833 |  0     | [-0.222, 0.222] | False         |

## 3. Nguồn chênh lệch

Chiều lệch (số nhãn / câu trả lời): useful: judge 1 – human 0 = 54, judge 0 – human 1 = 9; correct: judge 1 – human 0 = 16, judge 0 – human 1 = 1; hallucination: judge 1 – human 0 = 12, judge 0 – human 1 = 0

Answer correctness theo loại câu: multihop: human 0.222, judge 0.944 (13 câu judge 1 – human 0); single: human 0.611, judge 0.722 (3 câu judge 1 – human 0)

Passage hữu ích theo hệ: graphrag: human 0.153, judge 0.514; vector_rag: human 0.25, judge 0.552; theo vị trí: #1: human 0.389, judge 0.629; #2: human 0.167, judge 0.471; #3: human 0.083, judge 0.543; #4: human 0.167, judge 0.486

Faithfulness theo hệ: graphrag: human 0.991, judge 0.838 (9 câu judge có ý không được hỗ trợ – human không); vector_rag: human 0.933, judge 0.899 (3 câu judge có ý không được hỗ trợ – human không)
