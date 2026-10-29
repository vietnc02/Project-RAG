# Người chấm so với LLM-judge (nội dung 1–4)

100 câu trả lời chấm mới (nội dung 1, 3, 4) + 32 câu trả lời của nội dung 2 (human_judge.py); chỉ câu có đáp án mà hệ không refuse. Judge `llama3.1:8b`, điểm gộp 3 lần chấm.

## 1. Agreement ở mức câu trả lời

| nhóm       | nhãn                                       |   n_labels |   n_answers |   human_pos |   judge_pos |   agreement | agreement_ci95   |   kappa | kappa_ci95      | mức (Landis & Koch)   |
|:-----------|:-------------------------------------------|-----------:|------------:|------------:|------------:|------------:|:-----------------|--------:|:----------------|:----------------------|
| Y1         | Có claim không được hỗ trợ (hallucination) |         50 |          50 |       0.12  |       0.6   |       0.52  | [0.38, 0.66]     |   0.167 | [0.053, 0.317]  | rất thấp              |
| Y1         | Câu trả lời đúng (answer correctness)      |         50 |          50 |       0.84  |       0.96  |       0.88  | [0.78, 0.96]     |   0.359 | [0.0, 0.73]     | thấp                  |
| Y2         | Có claim không được hỗ trợ (hallucination) |         32 |          32 |       0.156 |       0.531 |       0.625 | [0.469, 0.781]   |   0.281 | [0.077, 0.548]  | thấp                  |
| Y2         | Câu trả lời đúng (answer correctness)      |         32 |          32 |       0.469 |       0.938 |       0.469 | [0.312, 0.625]   |  -0.007 | [-0.185, 0.158] | kém                   |
| Y3         | Có claim không được hỗ trợ (hallucination) |         25 |          25 |       0.32  |       0.72  |       0.52  | [0.32, 0.72]     |   0.171 | [-0.068, 0.468] | rất thấp              |
| Y3         | Câu trả lời đúng (answer correctness)      |         25 |          25 |       0.68  |       0.92  |       0.76  | [0.6, 0.92]      |   0.312 | [0.0, 0.706]    | thấp                  |
| Y4         | Có claim không được hỗ trợ (hallucination) |         25 |          25 |       0.16  |       0.56  |       0.52  | [0.32, 0.72]     |   0.112 | [-0.143, 0.398] | rất thấp              |
| Y4         | Câu trả lời đúng (answer correctness)      |         25 |          25 |       0.68  |       0.92  |       0.76  | [0.6, 0.92]      |   0.312 | [0.0, 0.658]    | thấp                  |
| Y1–Y4      | Có claim không được hỗ trợ (hallucination) |        132 |         132 |       0.174 |       0.598 |       0.545 | [0.462, 0.629]   |   0.194 | [0.098, 0.299]  | rất thấp              |
| Y1–Y4      | Câu trả lời đúng (answer correctness)      |        132 |         132 |       0.689 |       0.939 |       0.735 | [0.659, 0.803]   |   0.205 | [0.071, 0.353]  | thấp                  |
| Y1, Y3, Y4 | Có claim không được hỗ trợ (hallucination) |        100 |         100 |       0.18  |       0.62  |       0.52  | [0.42, 0.62]     |   0.168 | [0.058, 0.287]  | rất thấp              |
| Y1, Y3, Y4 | Câu trả lời đúng (answer correctness)      |        100 |         100 |       0.76  |       0.94  |       0.82  | [0.74, 0.89]     |   0.336 | [0.115, 0.555]  | thấp                  |

*CI 95%: bootstrap theo câu trả lời. κ theo Landis & Koch. Hallucination của người: câu trả lời có ít nhất một claim không được hỗ trợ (không có trong context hoặc mâu thuẫn context).*

## 2. Faithfulness

| nhóm       |   n |   mean_human |   mean_judge |   mae |   pearson |
|:-----------|----:|-------------:|-------------:|------:|----------:|
| Y1         |  50 |        0.957 |        0.81  | 0.155 |     0.492 |
| Y2         |  32 |        0.962 |        0.869 | 0.102 |     0.431 |
| Y3         |  25 |        0.908 |        0.743 | 0.191 |     0.315 |
| Y4         |  25 |        0.923 |        0.835 | 0.168 |     0.047 |
| Y1–Y4      | 132 |        0.942 |        0.816 | 0.152 |     0.347 |
| Y1, Y3, Y4 | 100 |        0.936 |        0.8   | 0.168 |     0.331 |

## 3. Y1.1: 40 câu single-hop của RAG thuần

Judge gắn hallucination 20 câu: người đồng ý 2, judge sai 18 (mọi claim được hỗ trợ); trong các câu người đồng ý, 0 câu chỉ có claim không có trong context, 2 câu có claim mâu thuẫn context. Judge không gắn 20 câu: người thấy có claim không được hỗ trợ ở 0 câu. Trên 180 claim (4.5 claim mỗi câu): 0 không có trong context, 3 mâu thuẫn context.
