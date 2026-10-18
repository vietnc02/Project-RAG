# Human judge so với LLM-judge (ý 2)

36 câu trả lời (18 câu hỏi × 2 hệ; single-hop 9, multi-hop 9 câu hỏi), judge gốc `llama3.1:8b` (results/kg30).

## 1. Mức agreement

| nhãn                                   |   n_labels |   n_answers |   human_pos |   judge_pos |   agreement | agreement_ci95   |   kappa | kappa_ci95      | mức (Landis & Koch)   |
|:---------------------------------------|-----------:|------------:|------------:|------------:|------------:|:-----------------|--------:|:----------------|:----------------------|
| Passage hữu ích (retrieval precision)  |        144 |          36 |       0.201 |       0.514 |       0.562 | [0.465, 0.66]    |   0.139 | [0.002, 0.289]  | rất thấp              |
| Câu trả lời đúng (answer correctness)  |         36 |          36 |       0.417 |       0.833 |       0.528 | [0.361, 0.694]   |   0.15  | [-0.056, 0.369] | rất thấp              |
| Có ý không được hỗ trợ (hallucination) |         32 |          32 |       0.156 |       0.531 |       0.625 | [0.469, 0.781]   |   0.281 | [0.077, 0.548]  | thấp                  |

*CI 95%: bootstrap theo câu trả lời. κ theo Landis & Koch: < 0,2 rất thấp, 0,2–0,4 thấp, 0,4–0,6 trung bình, 0,6–0,8 khá, > 0,8 gần như hoàn toàn. Hallucination chỉ tính câu không refuse; judge gốc không lưu verdict từng ý nên so ở mức câu trả lời. Khi một nhãn chiếm phần lớn (human_pos / judge_pos gần 0 hoặc 1, vd. answer correctness), κ có thể thấp dù % agreement cao (prevalence paradox), nên đọc κ cùng % agreement.*

Faithfulness (mức câu trả lời, n = 32): trung bình human judge 0.962, judge 0.869; sai khác tuyệt đối trung bình 0.102; tương quan Pearson 0.431.

## 2. GraphRAG − Vector RAG trên cùng mẫu: nhãn human judge và nhãn judge

| metric            | who   |   n_pairs |   graphrag |   vector_rag |   diff | ci95            | significant   |
|:------------------|:------|----------:|-----------:|-------------:|-------:|:----------------|:--------------|
| context_precision | human |        18 |      0.153 |        0.25  | -0.097 | [-0.208, 0.014] | False         |
| context_precision | judge |        18 |      0.514 |        0.514 |  0     | [-0.153, 0.153] | False         |
| faithfulness      | human |        15 |      0.99  |        0.928 |  0.062 | [0.01, 0.138]   | True          |
| faithfulness      | judge |        15 |      0.844 |        0.893 | -0.049 | [-0.145, 0.056] | False         |
| hallucination     | human |        18 |      0.056 |        0.222 | -0.167 | [-0.333, 0.0]   | False         |
| hallucination     | judge |        18 |      0.556 |        0.389 |  0.167 | [-0.111, 0.444] | False         |
| correct           | human |        18 |      0.333 |        0.5   | -0.167 | [-0.389, 0.056] | False         |
| correct           | judge |        18 |      0.833 |        0.833 |  0     | [-0.222, 0.222] | False         |
