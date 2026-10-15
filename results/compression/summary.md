# Kết quả ý 4: context compression cho GraphRAG

85 câu của ý 2 (62 câu có đáp án). LLM `qwen2.5:7b`, judge `llama3.1:8b`. Nén: KG_KEEP = 0.5, KG_ITEM_CHARS = 110, SENT_KEEP = 0.5. Version: `*/run_info.json`.

## 1. Token reduction và chất lượng (câu có đáp án; token reduction tính trên mọi câu)

| mode   |   n |   context_tokens |   token_reduction | token_reduction_ci95   |   prompt_reduction |   faithfulness | faithfulness_ci95   |   faithfulness_n |   hallucination |   answer_correctness |   context_recall |   answer_relevancy |   refusal_answerable | unanswerable_refused   |
|:-------|----:|-----------------:|------------------:|:-----------------------|-------------------:|---------------:|:--------------------|-----------------:|----------------:|---------------------:|-----------------:|-------------------:|---------------------:|:-----------------------|
| base   |  62 |           1539.3 |            0      | [0.000, 0.000]         |             0      |         0.864  | [0.825, 0.902]      |               57 |          0.4677 |               0.8065 |           0.6838 |             0.7552 |               0.0806 | 23/23                  |
| kg     |  62 |           1190.2 |            0.2254 | [0.217, 0.234]         |             0.2126 |         0.839  | [0.787, 0.888]      |               52 |          0.4839 |               0.7903 |           0.6899 |             0.6792 |               0.1613 | 23/23                  |
| text   |  62 |           1150.8 |            0.2533 | [0.246, 0.260]         |             0.239  |         0.8363 | [0.790, 0.878]      |               55 |          0.5645 |               0.7903 |           0.664  |             0.7374 |               0.1129 | 23/23                  |
| both   |  62 |            801.7 |            0.4786 | [0.470, 0.487]         |             0.4516 |         0.8147 | [0.754, 0.871]      |               52 |          0.4355 |               0.8065 |           0.6472 |             0.6933 |               0.1613 | 23/23                  |

*refusal_answerable: tỉ lệ refuse ở câu CÓ đáp án (câu refusal có answer_relevancy = 0, không có claim để tính faithfulness).*

## 2. Giữ nguyên faithfulness? (non-inferiority, ngưỡng −0.05)

| mode   |   n_pairs |    diff | ci95             |   margin | non_inferior   |
|:-------|----------:|--------:|:-----------------|---------:|:---------------|
| kg     |        52 | -0.0298 | [-0.079, 0.016]  |    -0.05 | False          |
| text   |        55 | -0.0288 | [-0.079, 0.016]  |    -0.05 | False          |
| both   |        52 | -0.0541 | [-0.110, -0.003] |    -0.05 | False          |

*non_inferior = True: cận dưới CI 95% của (nén − base) > −0,05, tức faithfulness không giảm quá 0,05. False: không đủ bằng chứng để nói faithfulness được giữ nguyên.*

## 3. Chênh lệch ghép cặp so với base (câu có đáp án)

| mode   | metric             |   n_pairs |    diff | ci95             | significant   |
|:-------|:-------------------|----------:|--------:|:-----------------|:--------------|
| kg     | faithfulness       |        52 | -0.0298 | [-0.079, 0.016]  | False         |
| kg     | hallucination      |        62 |  0.0161 | [-0.097, 0.145]  | False         |
| kg     | answer_correctness |        62 | -0.0161 | [-0.097, 0.065]  | False         |
| kg     | context_recall     |        62 |  0.0061 | [-0.026, 0.039]  | False         |
| kg     | answer_relevancy   |        62 | -0.076  | [-0.146, -0.021] | True          |
| kg     | faithfulness_full  |        52 | -0.0397 | [-0.088, 0.007]  | False         |
| kg     | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |
| text   | faithfulness       |        55 | -0.0288 | [-0.079, 0.016]  | False         |
| text   | hallucination      |        62 |  0.0968 | [-0.032, 0.226]  | False         |
| text   | answer_correctness |        62 | -0.0161 | [-0.081, 0.032]  | False         |
| text   | context_recall     |        62 | -0.0198 | [-0.054, 0.017]  | False         |
| text   | answer_relevancy   |        62 | -0.0178 | [-0.060, 0.013]  | False         |
| text   | faithfulness_full  |        55 | -0.0249 | [-0.071, 0.020]  | False         |
| text   | refusal            |        62 |  0.0323 | [0.000, 0.081]   | False         |
| both   | faithfulness       |        52 | -0.0541 | [-0.110, -0.003] | True          |
| both   | hallucination      |        62 | -0.0323 | [-0.145, 0.081]  | False         |
| both   | answer_correctness |        62 |  0      | [-0.097, 0.081]  | False         |
| both   | context_recall     |        62 | -0.0366 | [-0.071, -0.003] | True          |
| both   | answer_relevancy   |        62 | -0.0619 | [-0.127, -0.011] | True          |
| both   | faithfulness_full  |        52 | -0.0471 | [-0.104, 0.006]  | False         |
| both   | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |

## 4. Chỉ số phụ

| mode   |   faithfulness_full | faithfulness_full_ci95   |   hallucination_full |   latency_s |
|:-------|--------------------:|:-------------------------|---------------------:|------------:|
| base   |              0.864  | [0.825, 0.902]           |               0.4677 |       14.45 |
| kg     |              0.8291 | [0.777, 0.876]           |               0.5161 |       15.01 |
| text   |              0.8402 | [0.796, 0.879]           |               0.5484 |       14.44 |
| both   |              0.8218 | [0.764, 0.877]           |               0.4516 |        9.7  |

*faithfulness_full: claim của câu trả lời chấm trên context GỐC chưa nén.*

## 5. Judge failures và truncate

| mode         | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   prompt_tokens_max |   prompt_tokens_median |
|:-------------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|--------------------:|-----------------------:|
| base         | correctness       |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 360 |                    246 |
| base         | extract_claims    |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 219 |                    152 |
| base         | reverse_questions |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 218 |                    138 |
| base         | verify_claims     |     119 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2163 |                   1644 |
| base         | (tổng)            |     290 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2163 |                    257 |
| both         | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 327 |                    229 |
| both         | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 213 |                    149 |
| both         | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 193 |                    141 |
| both         | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2200 |                    940 |
| both         | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2200 |                    750 |
| kg           | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 328 |                    252 |
| kg           | extract_claims    |      52 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |                 207 |                    148 |
| kg           | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 206 |                    141 |
| kg           | verify_claims     |     166 |                1 |            0 |              0 |            0 |                        1 |                  1 |                  0 |                2190 |                   1342 |
| kg           | (tổng)            |     322 |                2 |            1 |              0 |            1 |                        1 |                  1 |                  0 |                2190 |                   1107 |
| text         | correctness       |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 374 |                    245 |
| text         | extract_claims    |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 233 |                    152 |
| text         | reverse_questions |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 215 |                    144 |
| text         | verify_claims     |     172 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2228 |                   1344 |
| text         | (tổng)            |     337 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2228 |                   1041 |
| ground truth | extract_claims    |      62 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 248 |                    164 |
| ground truth | (tổng)            |      62 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 248 |                    164 |
