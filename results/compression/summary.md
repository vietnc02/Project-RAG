# Kết quả ý 4: context compression cho GraphRAG

85 câu của ý 2 (62 câu có đáp án). LLM `qwen2.5:7b`, judge `llama3.1:8b`. Nén: KG_KEEP = 0.5, KG_ITEM_CHARS = 110, SENT_KEEP = 0.5. Version: `*/run_info.json`.

## 1. Token reduction và chất lượng (câu có đáp án; token reduction tính trên mọi câu)

| mode         |   n |   context_tokens |   token_reduction | token_reduction_ci95   |   prompt_reduction |   faithfulness | faithfulness_ci95   |   faithfulness_n |   hallucination |   answer_correctness |   context_recall |   answer_relevancy |   refusal_answerable | unanswerable_refused   |
|:-------------|----:|-----------------:|------------------:|:-----------------------|-------------------:|---------------:|:--------------------|-----------------:|----------------:|---------------------:|-----------------:|-------------------:|---------------------:|:-----------------------|
| base         |  62 |           1539.3 |            0      | [0.000, 0.000]         |             0      |         0.8557 | [0.817, 0.893]      |               57 |          0.4839 |               0.8065 |           0.6927 |             0.7512 |               0.0806 | 23/23                  |
| kg           |  62 |           1190.2 |            0.2254 | [0.217, 0.234]         |             0.2126 |         0.8418 | [0.795, 0.886]      |               52 |          0.4839 |               0.7742 |           0.6859 |             0.6786 |               0.1613 | 23/23                  |
| text         |  62 |           1150.8 |            0.2533 | [0.246, 0.260]         |             0.239  |         0.8326 | [0.787, 0.875]      |               55 |          0.5645 |               0.7903 |           0.6706 |             0.7398 |               0.1129 | 23/23                  |
| both         |  62 |            801.7 |            0.4786 | [0.470, 0.487]         |             0.4516 |         0.8016 | [0.743, 0.860]      |               52 |          0.4839 |               0.7903 |           0.6603 |             0.6901 |               0.1613 | 23/23                  |
| rand_both    |  62 |            793.9 |            0.4833 | [0.475, 0.492]         |             0.4561 |         0.806  | [0.752, 0.856]      |               49 |          0.4839 |               0.7097 |           0.638  |             0.6563 |               0.2097 | 23/23                  |
| rand_both_s1 |  62 |            791   |            0.4852 | [0.477, 0.494]         |             0.4578 |         0.8428 | [0.804, 0.879]      |               47 |          0.5    |               0.6935 |           0.6055 |             0.6205 |               0.2419 | 23/23                  |
| rand_both_s2 |  62 |            788.5 |            0.487  | [0.478, 0.496]         |             0.4595 |         0.8392 | [0.797, 0.881]      |               47 |          0.4677 |               0.7097 |           0.5926 |             0.6115 |               0.2419 | 23/23                  |
| head_both    |  62 |            796   |            0.4818 | [0.473, 0.491]         |             0.4546 |         0.7912 | [0.740, 0.840]      |               50 |          0.5484 |               0.7581 |           0.6002 |             0.6731 |               0.1935 | 23/23                  |
| rand_kg      |  62 |           1183.6 |            0.2297 | [0.222, 0.237]         |             0.2167 |         0.8622 | [0.822, 0.903]      |               52 |          0.4839 |               0.7903 |           0.7051 |             0.6812 |               0.1613 | 23/23                  |
| head_kg      |  62 |           1186.4 |            0.2278 | [0.220, 0.236]         |             0.215  |         0.855  | [0.812, 0.896]      |               52 |          0.4516 |               0.8065 |           0.6879 |             0.6887 |               0.1613 | 23/23                  |
| rand_text    |  62 |           1149.6 |            0.2534 | [0.246, 0.261]         |             0.2391 |         0.8245 | [0.775, 0.874]      |               54 |          0.4839 |               0.8065 |           0.6218 |             0.699  |               0.129  | 23/23                  |
| head_text    |  62 |           1148.9 |            0.2539 | [0.246, 0.262]         |             0.2397 |         0.8172 | [0.769, 0.863]      |               53 |          0.5323 |               0.7742 |           0.6236 |             0.6768 |               0.1452 | 23/23                  |
| both_l25     |  62 |           1125.3 |            0.2681 | [0.261, 0.276]         |             0.253  |         0.8359 | [0.786, 0.881]      |               54 |          0.5323 |               0.7903 |           0.6892 |             0.7143 |               0.129  | 23/23                  |
| both_l40     |  62 |            939.2 |            0.3894 | [0.382, 0.397]         |             0.3674 |         0.8179 | [0.773, 0.859]      |               52 |          0.5161 |               0.7903 |           0.6552 |             0.6864 |               0.1613 | 23/23                  |

*refusal_answerable: tỉ lệ refuse ở câu CÓ đáp án (câu refusal có answer_relevancy = 0, không có claim để tính faithfulness).*

## 2. Giữ nguyên faithfulness? (non-inferiority, ngưỡng −0.05)

| mode         |   n_pairs |    diff | ci95             |   margin | non_inferior   |
|:-------------|----------:|--------:|:-----------------|---------:|:---------------|
| kg           |        52 | -0.0257 | [-0.070, 0.018]  |    -0.05 | False          |
| text         |        55 | -0.0252 | [-0.074, 0.020]  |    -0.05 | False          |
| both         |        52 | -0.0659 | [-0.120, -0.016] |    -0.05 | False          |
| rand_both    |        49 | -0.0644 | [-0.117, -0.014] |    -0.05 | False          |
| rand_both_s1 |        47 | -0.0266 | [-0.067, 0.013]  |    -0.05 | False          |
| rand_both_s2 |        47 | -0.032  | [-0.079, 0.016]  |    -0.05 | False          |
| head_both    |        50 | -0.0704 | [-0.116, -0.027] |    -0.05 | False          |
| rand_kg      |        52 | -0.0136 | [-0.060, 0.033]  |    -0.05 | False          |
| head_kg      |        52 | -0.0158 | [-0.052, 0.021]  |    -0.05 | False          |
| rand_text    |        54 | -0.0399 | [-0.102, 0.019]  |    -0.05 | False          |
| head_text    |        53 | -0.0447 | [-0.097, 0.004]  |    -0.05 | False          |
| both_l25     |        54 | -0.023  | [-0.063, 0.015]  |    -0.05 | False          |
| both_l40     |        52 | -0.0497 | [-0.094, -0.010] |    -0.05 | False          |

*non_inferior = True: cận dưới CI 95% của (nén − base) > −0,05, tức faithfulness không giảm quá 0,05. False: không đủ bằng chứng để nói faithfulness được giữ nguyên.*

## 3. Chênh lệch ghép cặp so với base (câu có đáp án)

| mode         | metric             |   n_pairs |    diff | ci95             | significant   |
|:-------------|:-------------------|----------:|--------:|:-----------------|:--------------|
| kg           | faithfulness       |        52 | -0.0257 | [-0.070, 0.018]  | False         |
| kg           | hallucination      |        62 |  0      | [-0.129, 0.129]  | False         |
| kg           | answer_correctness |        62 | -0.0323 | [-0.113, 0.048]  | False         |
| kg           | context_recall     |        62 | -0.0068 | [-0.037, 0.023]  | False         |
| kg           | answer_relevancy   |        62 | -0.0727 | [-0.143, -0.018] | True          |
| kg           | faithfulness_full  |        52 | -0.032  | [-0.079, 0.011]  | False         |
| kg           | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |
| text         | faithfulness       |        55 | -0.0252 | [-0.074, 0.020]  | False         |
| text         | hallucination      |        62 |  0.0806 | [-0.048, 0.210]  | False         |
| text         | answer_correctness |        62 | -0.0161 | [-0.081, 0.032]  | False         |
| text         | context_recall     |        62 | -0.0221 | [-0.057, 0.012]  | False         |
| text         | answer_relevancy   |        62 | -0.0114 | [-0.054, 0.019]  | False         |
| text         | faithfulness_full  |        55 | -0.0243 | [-0.068, 0.016]  | False         |
| text         | refusal            |        62 |  0.0323 | [0.000, 0.081]   | False         |
| both         | faithfulness       |        52 | -0.0659 | [-0.120, -0.016] | True          |
| both         | hallucination      |        62 |  0      | [-0.113, 0.113]  | False         |
| both         | answer_correctness |        62 | -0.0161 | [-0.097, 0.065]  | False         |
| both         | context_recall     |        62 | -0.0324 | [-0.066, 0.003]  | False         |
| both         | answer_relevancy   |        62 | -0.0611 | [-0.127, -0.011] | True          |
| both         | faithfulness_full  |        52 | -0.0378 | [-0.093, 0.012]  | False         |
| both         | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |
| rand_both    | faithfulness       |        49 | -0.0644 | [-0.117, -0.014] | True          |
| rand_both    | hallucination      |        62 |  0      | [-0.145, 0.145]  | False         |
| rand_both    | answer_correctness |        62 | -0.0968 | [-0.194, 0.000]  | False         |
| rand_both    | context_recall     |        61 | -0.0551 | [-0.092, -0.019] | True          |
| rand_both    | answer_relevancy   |        62 | -0.0949 | [-0.174, -0.025] | True          |
| rand_both    | faithfulness_full  |        49 | -0.0598 | [-0.105, -0.017] | True          |
| rand_both    | refusal            |        62 |  0.129  | [0.048, 0.210]   | True          |
| rand_both_s1 | faithfulness       |        47 | -0.0266 | [-0.067, 0.013]  | False         |
| rand_both_s1 | hallucination      |        62 |  0.0161 | [-0.129, 0.161]  | False         |
| rand_both_s1 | answer_correctness |        62 | -0.1129 | [-0.226, 0.000]  | False         |
| rand_both_s1 | context_recall     |        62 | -0.0872 | [-0.128, -0.047] | True          |
| rand_both_s1 | answer_relevancy   |        62 | -0.1308 | [-0.212, -0.058] | True          |
| rand_both_s1 | faithfulness_full  |        47 | -0.0085 | [-0.051, 0.034]  | False         |
| rand_both_s1 | refusal            |        62 |  0.1613 | [0.081, 0.258]   | True          |
| rand_both_s2 | faithfulness       |        47 | -0.032  | [-0.079, 0.016]  | False         |
| rand_both_s2 | hallucination      |        62 | -0.0161 | [-0.177, 0.145]  | False         |
| rand_both_s2 | answer_correctness |        62 | -0.0968 | [-0.194, 0.000]  | False         |
| rand_both_s2 | context_recall     |        61 | -0.1006 | [-0.140, -0.060] | True          |
| rand_both_s2 | answer_relevancy   |        62 | -0.1398 | [-0.224, -0.065] | True          |
| rand_both_s2 | faithfulness_full  |        47 | -0.0132 | [-0.059, 0.032]  | False         |
| rand_both_s2 | refusal            |        62 |  0.1613 | [0.081, 0.258]   | True          |
| head_both    | faithfulness       |        50 | -0.0704 | [-0.116, -0.027] | True          |
| head_both    | hallucination      |        62 |  0.0645 | [-0.048, 0.177]  | False         |
| head_both    | answer_correctness |        62 | -0.0484 | [-0.145, 0.048]  | False         |
| head_both    | context_recall     |        62 | -0.0925 | [-0.141, -0.047] | True          |
| head_both    | answer_relevancy   |        62 | -0.0782 | [-0.152, -0.016] | True          |
| head_both    | faithfulness_full  |        50 | -0.0337 | [-0.072, 0.003]  | False         |
| head_both    | refusal            |        62 |  0.1129 | [0.048, 0.194]   | True          |
| rand_kg      | faithfulness       |        52 | -0.0136 | [-0.060, 0.033]  | False         |
| rand_kg      | hallucination      |        62 |  0      | [-0.161, 0.161]  | False         |
| rand_kg      | answer_correctness |        62 | -0.0161 | [-0.097, 0.065]  | False         |
| rand_kg      | context_recall     |        62 |  0.0124 | [-0.017, 0.041]  | False         |
| rand_kg      | answer_relevancy   |        62 | -0.07   | [-0.141, -0.011] | True          |
| rand_kg      | faithfulness_full  |        52 | -0.0161 | [-0.060, 0.027]  | False         |
| rand_kg      | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |
| head_kg      | faithfulness       |        52 | -0.0158 | [-0.052, 0.021]  | False         |
| head_kg      | hallucination      |        62 | -0.0323 | [-0.161, 0.097]  | False         |
| head_kg      | answer_correctness |        62 |  0      | [-0.081, 0.081]  | False         |
| head_kg      | context_recall     |        62 | -0.0048 | [-0.038, 0.027]  | False         |
| head_kg      | answer_relevancy   |        62 | -0.0625 | [-0.124, -0.009] | True          |
| head_kg      | faithfulness_full  |        52 | -0.0071 | [-0.041, 0.028]  | False         |
| head_kg      | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |
| rand_text    | faithfulness       |        54 | -0.0399 | [-0.102, 0.019]  | False         |
| rand_text    | hallucination      |        62 |  0      | [-0.161, 0.145]  | False         |
| rand_text    | answer_correctness |        62 |  0      | [-0.048, 0.048]  | False         |
| rand_text    | context_recall     |        62 | -0.0709 | [-0.110, -0.034] | True          |
| rand_text    | answer_relevancy   |        62 | -0.0523 | [-0.110, -0.004] | True          |
| rand_text    | faithfulness_full  |        54 | -0.0149 | [-0.069, 0.035]  | False         |
| rand_text    | refusal            |        62 |  0.0484 | [0.000, 0.113]   | False         |
| head_text    | faithfulness       |        53 | -0.0447 | [-0.097, 0.004]  | False         |
| head_text    | hallucination      |        62 |  0.0484 | [-0.081, 0.177]  | False         |
| head_text    | answer_correctness |        62 | -0.0323 | [-0.113, 0.048]  | False         |
| head_text    | context_recall     |        62 | -0.0691 | [-0.114, -0.029] | True          |
| head_text    | answer_relevancy   |        62 | -0.0744 | [-0.141, -0.016] | True          |
| head_text    | faithfulness_full  |        53 | -0.021  | [-0.060, 0.017]  | False         |
| head_text    | refusal            |        62 |  0.0645 | [0.016, 0.129]   | True          |
| both_l25     | faithfulness       |        54 | -0.023  | [-0.063, 0.015]  | False         |
| both_l25     | hallucination      |        62 |  0.0484 | [-0.081, 0.177]  | False         |
| both_l25     | answer_correctness |        62 | -0.0161 | [-0.081, 0.048]  | False         |
| both_l25     | context_recall     |        62 | -0.0035 | [-0.033, 0.025]  | False         |
| both_l25     | answer_relevancy   |        62 | -0.0369 | [-0.087, 0.003]  | False         |
| both_l25     | faithfulness_full  |        54 | -0.0242 | [-0.065, 0.014]  | False         |
| both_l25     | refusal            |        62 |  0.0484 | [0.000, 0.113]   | False         |
| both_l40     | faithfulness       |        52 | -0.0497 | [-0.094, -0.010] | True          |
| both_l40     | hallucination      |        62 |  0.0323 | [-0.113, 0.177]  | False         |
| both_l40     | answer_correctness |        62 | -0.0161 | [-0.113, 0.081]  | False         |
| both_l40     | context_recall     |        62 | -0.0375 | [-0.070, -0.006] | True          |
| both_l40     | answer_relevancy   |        62 | -0.0648 | [-0.129, -0.015] | True          |
| both_l40     | faithfulness_full  |        52 | -0.0347 | [-0.077, 0.005]  | False         |
| both_l40     | refusal            |        62 |  0.0806 | [0.016, 0.161]   | True          |

## 4. Chỉ số phụ

| mode         |   faithfulness_full | faithfulness_full_ci95   |   hallucination_full |   latency_s |
|:-------------|--------------------:|:-------------------------|---------------------:|------------:|
| base         |              0.8557 | [0.817, 0.893]           |               0.4839 |       14.45 |
| kg           |              0.8356 | [0.787, 0.883]           |               0.4839 |       15.01 |
| text         |              0.8335 | [0.789, 0.874]           |               0.5484 |       14.44 |
| both         |              0.8298 | [0.771, 0.885]           |               0.4516 |        9.7  |
| rand_both    |              0.8106 | [0.763, 0.854]           |               0.5161 |       12.22 |
| rand_both_s1 |              0.8609 | [0.819, 0.902]           |               0.4032 |        9.12 |
| rand_both_s2 |              0.8581 | [0.820, 0.893]           |               0.4516 |        9.27 |
| head_both    |              0.8279 | [0.783, 0.870]           |               0.5323 |        9.31 |
| rand_kg      |              0.8597 | [0.820, 0.899]           |               0.4839 |       12.52 |
| head_kg      |              0.8636 | [0.822, 0.903]           |               0.4516 |       11.95 |
| rand_text    |              0.8496 | [0.805, 0.891]           |               0.4677 |       11.7  |
| head_text    |              0.8409 | [0.800, 0.882]           |               0.4839 |        9.99 |
| both_l25     |              0.8347 | [0.786, 0.880]           |               0.5161 |       15.22 |
| both_l40     |              0.8328 | [0.787, 0.874]           |               0.5    |       13.75 |

*faithfulness_full: claim của câu trả lời chấm trên context GỐC chưa nén.*

## 5. Judge failures và truncate

| mode            | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   retries |   unresolved |   prompt_tokens_max |   prompt_tokens_median |
|:----------------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|----------:|-------------:|--------------------:|-----------------------:|
| base.p0         | correctness       |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 360 |                    246 |
| base.p0         | extract_claims    |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 219 |                    152 |
| base.p0         | reverse_questions |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 218 |                    138 |
| base.p0         | verify_claims     |     119 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2163 |                   1644 |
| base.p0         | (tổng)            |     290 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2163 |                    257 |
| base.p1         | correctness       |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 360 |                    246 |
| base.p1         | extract_claims    |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 219 |                    152 |
| base.p1         | reverse_questions |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 218 |                    138 |
| base.p1         | verify_claims     |     119 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2163 |                   1644 |
| base.p1         | (tổng)            |     290 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2163 |                    257 |
| base.p2         | correctness       |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 360 |                    246 |
| base.p2         | extract_claims    |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 219 |                    152 |
| base.p2         | reverse_questions |      57 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 218 |                    138 |
| base.p2         | verify_claims     |     119 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2163 |                   1644 |
| base.p2         | (tổng)            |     290 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2163 |                    257 |
| both.p0         | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 327 |                    229 |
| both.p0         | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 213 |                    149 |
| both.p0         | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 193 |                    141 |
| both.p0         | verify_claims     |     167 |                1 |            1 |              0 |            1 |                        1 |                  6 |                  0 |         1 |            0 |                2200 |                    941 |
| both.p0         | (tổng)            |     323 |                1 |            1 |              0 |            1 |                        1 |                  6 |                  0 |         1 |            0 |                2200 |                    752 |
| both.p1         | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 327 |                    229 |
| both.p1         | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 213 |                    149 |
| both.p1         | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 193 |                    141 |
| both.p1         | verify_claims     |     167 |                1 |            1 |              0 |            1 |                        1 |                  6 |                  0 |         1 |            0 |                2200 |                    941 |
| both.p1         | (tổng)            |     323 |                1 |            1 |              0 |            1 |                        1 |                  6 |                  0 |         1 |            0 |                2200 |                    752 |
| both.p2         | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 327 |                    229 |
| both.p2         | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 213 |                    149 |
| both.p2         | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 193 |                    141 |
| both.p2         | verify_claims     |     167 |                1 |            1 |              0 |            1 |                        1 |                  6 |                  0 |         1 |            0 |                2200 |                    941 |
| both.p2         | (tổng)            |     323 |                1 |            1 |              0 |            1 |                        1 |                  6 |                  0 |         1 |            0 |                2200 |                    752 |
| both_l25.p0     | correctness       |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 383 |                    245 |
| both_l25.p0     | extract_claims    |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 242 |                    154 |
| both_l25.p0     | reverse_questions |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 224 |                    142 |
| both_l25.p0     | verify_claims     |     171 |                1 |            0 |              1 |            0 |                        1 |                  4 |                  0 |         1 |            0 |                2232 |                   1261 |
| both_l25.p0     | (tổng)            |     333 |                1 |            0 |              1 |            0 |                        1 |                  4 |                  0 |         1 |            0 |                2232 |                   1036 |
| both_l25.p1     | correctness       |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 383 |                    245 |
| both_l25.p1     | extract_claims    |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 242 |                    154 |
| both_l25.p1     | reverse_questions |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 224 |                    142 |
| both_l25.p1     | verify_claims     |     171 |                1 |            0 |              1 |            0 |                        1 |                  4 |                  0 |         1 |            0 |                2232 |                   1261 |
| both_l25.p1     | (tổng)            |     333 |                1 |            0 |              1 |            0 |                        1 |                  4 |                  0 |         1 |            0 |                2232 |                   1036 |
| both_l25.p2     | correctness       |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 383 |                    245 |
| both_l25.p2     | extract_claims    |      57 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                 242 |                    157 |
| both_l25.p2     | reverse_questions |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 224 |                    142 |
| both_l25.p2     | verify_claims     |     172 |                2 |            0 |              2 |            0 |                        2 |                  7 |                  0 |         2 |            0 |                2232 |                   1266 |
| both_l25.p2     | (tổng)            |     337 |                6 |            4 |              2 |            4 |                        2 |                  7 |                  0 |         5 |            1 |                2232 |                   1034 |
| both_l40.p0     | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 345 |                    239 |
| both_l40.p0     | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 204 |                    152 |
| both_l40.p0     | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 186 |                    145 |
| both_l40.p0     | verify_claims     |     167 |                1 |            1 |              0 |            1 |                        1 |                  5 |                  0 |         1 |            0 |                2184 |                   1086 |
| both_l40.p0     | (tổng)            |     323 |                1 |            1 |              0 |            1 |                        1 |                  5 |                  0 |         1 |            0 |                2184 |                    876 |
| both_l40.p1     | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 345 |                    239 |
| both_l40.p1     | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 204 |                    152 |
| both_l40.p1     | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 186 |                    145 |
| both_l40.p1     | verify_claims     |     167 |                1 |            1 |              0 |            1 |                        1 |                  5 |                  0 |         1 |            0 |                2184 |                   1086 |
| both_l40.p1     | (tổng)            |     323 |                1 |            1 |              0 |            1 |                        1 |                  5 |                  0 |         1 |            0 |                2184 |                    876 |
| both_l40.p2     | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 345 |                    239 |
| both_l40.p2     | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 204 |                    152 |
| both_l40.p2     | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 186 |                    145 |
| both_l40.p2     | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2184 |                   1085 |
| both_l40.p2     | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2184 |                    878 |
| head_both.p0    | correctness       |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 352 |                    250 |
| head_both.p0    | extract_claims    |      51 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                 229 |                    148 |
| head_both.p0    | reverse_questions |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 212 |                    140 |
| head_both.p0    | verify_claims     |     162 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2156 |                    934 |
| head_both.p0    | (tổng)            |     313 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                2156 |                    765 |
| head_both.p1    | correctness       |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 352 |                    250 |
| head_both.p1    | extract_claims    |      51 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                 229 |                    148 |
| head_both.p1    | reverse_questions |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 212 |                    140 |
| head_both.p1    | verify_claims     |     162 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2156 |                    934 |
| head_both.p1    | (tổng)            |     313 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                2156 |                    765 |
| head_both.p2    | correctness       |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 352 |                    250 |
| head_both.p2    | extract_claims    |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 229 |                    148 |
| head_both.p2    | reverse_questions |      50 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 212 |                    140 |
| head_both.p2    | verify_claims     |     162 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2156 |                    925 |
| head_both.p2    | (tổng)            |     312 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2156 |                    764 |
| head_kg.p0      | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 324 |                    245 |
| head_kg.p0      | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 199 |                    152 |
| head_kg.p0      | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 181 |                    143 |
| head_kg.p0      | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1335 |
| head_kg.p0      | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1106 |
| head_kg.p1      | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 324 |                    245 |
| head_kg.p1      | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 199 |                    152 |
| head_kg.p1      | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 181 |                    143 |
| head_kg.p1      | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1335 |
| head_kg.p1      | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1106 |
| head_kg.p2      | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 324 |                    245 |
| head_kg.p2      | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 199 |                    152 |
| head_kg.p2      | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 181 |                    143 |
| head_kg.p2      | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1335 |
| head_kg.p2      | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1106 |
| head_text.p0    | correctness       |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 351 |                    241 |
| head_text.p0    | extract_claims    |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 223 |                    149 |
| head_text.p0    | reverse_questions |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 188 |                    140 |
| head_text.p0    | verify_claims     |     171 |                4 |            0 |              4 |            0 |                        4 |                 20 |                  0 |         3 |            1 |                2196 |                   1354 |
| head_text.p0    | (tổng)            |     330 |                4 |            0 |              4 |            0 |                        4 |                 20 |                  0 |         3 |            1 |                2196 |                   1076 |
| head_text.p1    | correctness       |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 351 |                    241 |
| head_text.p1    | extract_claims    |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 223 |                    149 |
| head_text.p1    | reverse_questions |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 188 |                    140 |
| head_text.p1    | verify_claims     |     171 |                4 |            0 |              4 |            0 |                        4 |                 20 |                  0 |         3 |            1 |                2196 |                   1354 |
| head_text.p1    | (tổng)            |     330 |                4 |            0 |              4 |            0 |                        4 |                 20 |                  0 |         3 |            1 |                2196 |                   1076 |
| head_text.p2    | correctness       |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 351 |                    241 |
| head_text.p2    | extract_claims    |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 223 |                    149 |
| head_text.p2    | reverse_questions |      53 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 188 |                    140 |
| head_text.p2    | verify_claims     |     171 |                4 |            0 |              4 |            0 |                        4 |                 16 |                  0 |         3 |            1 |                2196 |                   1349 |
| head_text.p2    | (tổng)            |     330 |                4 |            0 |              4 |            0 |                        4 |                 16 |                  0 |         3 |            1 |                2196 |                   1076 |
| kg.p0           | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 328 |                    252 |
| kg.p0           | extract_claims    |      55 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                 207 |                    150 |
| kg.p0           | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 206 |                    141 |
| kg.p0           | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1339 |
| kg.p0           | (tổng)            |     325 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                2190 |                   1086 |
| kg.p1           | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 328 |                    252 |
| kg.p1           | extract_claims    |      55 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                 207 |                    150 |
| kg.p1           | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 206 |                    141 |
| kg.p1           | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1339 |
| kg.p1           | (tổng)            |     325 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                2190 |                   1086 |
| kg.p2           | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 328 |                    252 |
| kg.p2           | extract_claims    |      55 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                 207 |                    150 |
| kg.p2           | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 206 |                    141 |
| kg.p2           | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2190 |                   1339 |
| kg.p2           | (tổng)            |     325 |                4 |            4 |              0 |            4 |                        0 |                  0 |                  0 |         3 |            1 |                2190 |                   1086 |
| rand_both.p0    | correctness       |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 325 |                    248 |
| rand_both.p0    | extract_claims    |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 200 |                    158 |
| rand_both.p0    | reverse_questions |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 183 |                    146 |
| rand_both.p0    | verify_claims     |     166 |                8 |            0 |              4 |            0 |                        8 |                 24 |                  0 |         6 |            2 |                2154 |                    965 |
| rand_both.p0    | (tổng)            |     313 |                8 |            0 |              4 |            0 |                        8 |                 24 |                  0 |         6 |            2 |                2154 |                    751 |
| rand_both.p1    | correctness       |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 325 |                    248 |
| rand_both.p1    | extract_claims    |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 200 |                    158 |
| rand_both.p1    | reverse_questions |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 183 |                    146 |
| rand_both.p1    | verify_claims     |     166 |                8 |            0 |              4 |            0 |                        8 |                 24 |                  0 |         6 |            2 |                2154 |                    965 |
| rand_both.p1    | (tổng)            |     313 |                8 |            0 |              4 |            0 |                        8 |                 24 |                  0 |         6 |            2 |                2154 |                    751 |
| rand_both.p2    | correctness       |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 325 |                    248 |
| rand_both.p2    | extract_claims    |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 200 |                    158 |
| rand_both.p2    | reverse_questions |      49 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 183 |                    146 |
| rand_both.p2    | verify_claims     |     164 |                5 |            0 |              4 |            0 |                        5 |                 15 |                  0 |         4 |            1 |                2154 |                    961 |
| rand_both.p2    | (tổng)            |     311 |                5 |            0 |              4 |            0 |                        5 |                 15 |                  0 |         4 |            1 |                2154 |                    751 |
| rand_both_s1.p0 | correctness       |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 330 |                    230 |
| rand_both_s1.p0 | extract_claims    |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 195 |                    146 |
| rand_both_s1.p0 | reverse_questions |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 177 |                    139 |
| rand_both_s1.p0 | verify_claims     |     156 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2176 |                    942 |
| rand_both_s1.p0 | (tổng)            |     297 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2176 |                    768 |
| rand_both_s1.p1 | correctness       |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 330 |                    230 |
| rand_both_s1.p1 | extract_claims    |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 195 |                    146 |
| rand_both_s1.p1 | reverse_questions |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 177 |                    139 |
| rand_both_s1.p1 | verify_claims     |     156 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2176 |                    942 |
| rand_both_s1.p1 | (tổng)            |     297 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2176 |                    768 |
| rand_both_s1.p2 | correctness       |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 330 |                    230 |
| rand_both_s1.p2 | extract_claims    |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 195 |                    146 |
| rand_both_s1.p2 | reverse_questions |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 177 |                    139 |
| rand_both_s1.p2 | verify_claims     |     162 |                8 |            0 |              8 |            0 |                        8 |                 20 |                  0 |         6 |            2 |                2176 |                    942 |
| rand_both_s1.p2 | (tổng)            |     303 |                8 |            0 |              8 |            0 |                        8 |                 20 |                  0 |         6 |            2 |                2176 |                    764 |
| rand_both_s2.p0 | correctness       |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 358 |                    246 |
| rand_both_s2.p0 | extract_claims    |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 223 |                    152 |
| rand_both_s2.p0 | reverse_questions |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 205 |                    145 |
| rand_both_s2.p0 | verify_claims     |     159 |                4 |            0 |              4 |            0 |                        4 |                 12 |                  0 |         3 |            1 |                2152 |                    933 |
| rand_both_s2.p0 | (tổng)            |     300 |                4 |            0 |              4 |            0 |                        4 |                 12 |                  0 |         3 |            1 |                2152 |                    757 |
| rand_both_s2.p1 | correctness       |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 358 |                    246 |
| rand_both_s2.p1 | extract_claims    |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 223 |                    152 |
| rand_both_s2.p1 | reverse_questions |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 205 |                    145 |
| rand_both_s2.p1 | verify_claims     |     159 |                4 |            0 |              4 |            0 |                        4 |                 12 |                  0 |         3 |            1 |                2152 |                    933 |
| rand_both_s2.p1 | (tổng)            |     300 |                4 |            0 |              4 |            0 |                        4 |                 12 |                  0 |         3 |            1 |                2152 |                    757 |
| rand_both_s2.p2 | correctness       |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 358 |                    246 |
| rand_both_s2.p2 | extract_claims    |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 223 |                    152 |
| rand_both_s2.p2 | reverse_questions |      47 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 205 |                    145 |
| rand_both_s2.p2 | verify_claims     |     159 |                4 |            0 |              4 |            0 |                        4 |                 12 |                  0 |         3 |            1 |                2152 |                    939 |
| rand_both_s2.p2 | (tổng)            |     300 |                4 |            0 |              4 |            0 |                        4 |                 12 |                  0 |         3 |            1 |                2152 |                    757 |
| rand_kg.p0      | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 339 |                    241 |
| rand_kg.p0      | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 208 |                    148 |
| rand_kg.p0      | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 188 |                    140 |
| rand_kg.p0      | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2153 |                   1324 |
| rand_kg.p0      | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2153 |                   1118 |
| rand_kg.p1      | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 339 |                    241 |
| rand_kg.p1      | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 208 |                    148 |
| rand_kg.p1      | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 188 |                    140 |
| rand_kg.p1      | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2153 |                   1324 |
| rand_kg.p1      | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2153 |                   1118 |
| rand_kg.p2      | correctness       |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 339 |                    241 |
| rand_kg.p2      | extract_claims    |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 208 |                    148 |
| rand_kg.p2      | reverse_questions |      52 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 188 |                    140 |
| rand_kg.p2      | verify_claims     |     166 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2153 |                   1324 |
| rand_kg.p2      | (tổng)            |     322 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2153 |                   1115 |
| rand_text.p0    | correctness       |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 345 |                    236 |
| rand_text.p0    | extract_claims    |      55 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                 216 |                    151 |
| rand_text.p0    | reverse_questions |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 186 |                    141 |
| rand_text.p0    | verify_claims     |     171 |                1 |            0 |              0 |            0 |                        1 |                  1 |                  0 |         1 |            0 |                2228 |                   1341 |
| rand_text.p0    | (tổng)            |     334 |                2 |            1 |              0 |            1 |                        1 |                  1 |                  0 |         2 |            0 |                2228 |                   1056 |
| rand_text.p1    | correctness       |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 345 |                    236 |
| rand_text.p1    | extract_claims    |      55 |                1 |            1 |              0 |            1 |                        0 |                  0 |                  0 |         1 |            0 |                 216 |                    151 |
| rand_text.p1    | reverse_questions |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 186 |                    141 |
| rand_text.p1    | verify_claims     |     171 |                1 |            0 |              0 |            0 |                        1 |                  1 |                  0 |         1 |            0 |                2228 |                   1341 |
| rand_text.p1    | (tổng)            |     334 |                2 |            1 |              0 |            1 |                        1 |                  1 |                  0 |         2 |            0 |                2228 |                   1056 |
| rand_text.p2    | correctness       |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 345 |                    236 |
| rand_text.p2    | extract_claims    |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 216 |                    151 |
| rand_text.p2    | reverse_questions |      54 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 186 |                    141 |
| rand_text.p2    | verify_claims     |     173 |                4 |            0 |              4 |            0 |                        4 |                  8 |                  0 |         3 |            1 |                2228 |                   1341 |
| rand_text.p2    | (tổng)            |     335 |                4 |            0 |              4 |            0 |                        4 |                  8 |                  0 |         3 |            1 |                2228 |                   1051 |
| text.p0         | correctness       |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 374 |                    245 |
| text.p0         | extract_claims    |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 233 |                    152 |
| text.p0         | reverse_questions |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 215 |                    144 |
| text.p0         | verify_claims     |     172 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2228 |                   1344 |
| text.p0         | (tổng)            |     337 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2228 |                   1041 |
| text.p1         | correctness       |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 374 |                    245 |
| text.p1         | extract_claims    |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 233 |                    152 |
| text.p1         | reverse_questions |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 215 |                    144 |
| text.p1         | verify_claims     |     172 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2228 |                   1344 |
| text.p1         | (tổng)            |     337 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2228 |                   1041 |
| text.p2         | correctness       |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 374 |                    245 |
| text.p2         | extract_claims    |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 233 |                    152 |
| text.p2         | reverse_questions |      55 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 215 |                    144 |
| text.p2         | verify_claims     |     172 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2228 |                   1344 |
| text.p2         | (tổng)            |     337 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                2228 |                   1041 |
| ground truth    | extract_claims    |      62 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 248 |                    164 |
| ground truth    | (tổng)            |      62 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |         0 |            0 |                 248 |                    164 |
