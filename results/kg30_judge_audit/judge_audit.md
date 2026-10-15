# Judge audit ý 2: re-judge results/kg30 với judge log

Judge `llama3.1:8b`, num_ctx = 4096, num_predict (JSON) = 1024. Ollama gốc 0.35.1, re-judge 0.40.4 (digest model: 46e0c10c039e). Answer giữ nguyên, chỉ chấm lại.

## 1. Judge failures và truncate (theo prompt)

| system     | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   prompt_tokens_max |   prompt_tokens_median |
|:-----------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|--------------------:|-----------------------:|
| vector_rag | correctness       |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 416 |                    248 |
| vector_rag | extract_claims    |     118 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 251 |                    153 |
| vector_rag | reverse_questions |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 236 |                    142 |
| vector_rag | useful            |      62 |                2 |            1 |              0 |            1 |                        2 |                  5 |                  0 |                1486 |                   1020 |
| vector_rag | verify_claims     |     118 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                1465 |                    999 |
| vector_rag | (tổng)            |     410 |                2 |            1 |              0 |            1 |                        2 |                  5 |                  0 |                1486 |                    256 |
| graphrag   | correctness       |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 361 |                    247 |
| graphrag   | extract_claims    |     118 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 248 |                    155 |
| graphrag   | reverse_questions |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 218 |                    139 |
| graphrag   | useful            |     124 |                5 |            0 |              5 |            0 |                        5 |                 44 |                  0 |                1545 |                    836 |
| graphrag   | verify_claims     |     174 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2163 |                   1573 |
| graphrag   | (tổng)            |     528 |                5 |            0 |              5 |            0 |                        5 |                 44 |                  0 |                2163 |                    566 |

- Tổng số judge call: 938; judge_failures: 7 (json_error: 1, ollama_error: 5, missing_verdicts: 7 call / 49 verdict); output_cut: 1.
- Prompt dài nhất: 2163 token (num_ctx = 4096); truncated prompt: 0. Ollama truncate prompt dài hơn num_ctx mà không báo lỗi, chỉ giữ đúng num_ctx / 2 + 2 = 2050 token, nên prompt_eval_count bằng số này (hoặc >= num_ctx) được đánh dấu truncated.

## 2. Reproducibility (điểm gốc so với re-judge, trên cùng answer)

| system     | metric             |   n |   mean_original |   mean_rejudged |   exact_agreement |   mean_abs_diff |   cohen_kappa |
|:-----------|:-------------------|----:|----------------:|----------------:|------------------:|----------------:|--------------:|
| vector_rag | context_precision  |  62 |          0.6169 |          0.6169 |            1      |          0      |           nan |
| vector_rag | faithfulness       |  56 |          0.8909 |          0.8909 |            1      |          0      |           nan |
| vector_rag | answer_relevancy   |  85 |          0.5422 |          0.5417 |            0.9882 |          0.0005 |           nan |
| vector_rag | hallucination      |  85 |          0.3176 |          0.3176 |            1      |        nan      |             1 |
| vector_rag | answer_correctness |  85 |          0.8941 |          0.8941 |            1      |        nan      |             1 |
| vector_rag | refusal            |  85 |          0.3412 |          0.3412 |            1      |        nan      |             1 |
| graphrag   | context_precision  |  62 |          0.4879 |          0.4879 |            1      |          0      |           nan |
| graphrag   | faithfulness       |  56 |          0.8468 |          0.8468 |            1      |          0      |           nan |
| graphrag   | answer_relevancy   |  85 |          0.5401 |          0.5398 |            0.9765 |          0.0002 |           nan |
| graphrag   | hallucination      |  85 |          0.3882 |          0.3882 |            1      |        nan      |             1 |
| graphrag   | answer_correctness |  85 |          0.8706 |          0.8706 |            1      |        nan      |             1 |
| graphrag   | refusal            |  85 |          0.3412 |          0.3412 |            1      |        nan      |             1 |

*exact_agreement: tỉ lệ câu có điểm giống hệt; cohen_kappa cho metric nhị phân; mean_abs_diff cho metric liên tục. faithfulness chỉ tính trên câu không refusal.*

## 3. Paired difference GraphRAG − Vector RAG: điểm gốc và re-judge

| type         | metric             |   diff_original | ci95_original    | significant_original   |   diff_rejudged | ci95_rejudged    | significant_rejudged   |
|:-------------|:-------------------|----------------:|:-----------------|:-----------------------|----------------:|:-----------------|:-----------------------|
| all          | context_precision  |         -0.129  | [-0.206, -0.048] | True                   |         -0.129  | [-0.206, -0.048] | True                   |
| all          | faithfulness       |         -0.0303 | [-0.080, 0.020]  | False                  |         -0.0303 | [-0.080, 0.020]  | False                  |
| all          | answer_relevancy   |         -0.003  | [-0.090, 0.082]  | False                  |         -0.0025 | [-0.090, 0.083]  | False                  |
| all          | hallucination_rate |          0.0968 | [-0.048, 0.226]  | False                  |          0.0968 | [-0.048, 0.226]  | False                  |
| all          | answer_correctness |         -0.0323 | [-0.145, 0.081]  | False                  |         -0.0323 | [-0.145, 0.081]  | False                  |
| single       | context_precision  |         -0.0556 | [-0.176, 0.074]  | False                  |         -0.0556 | [-0.176, 0.074]  | False                  |
| single       | faithfulness       |         -0.0273 | [-0.098, 0.047]  | False                  |         -0.0273 | [-0.098, 0.047]  | False                  |
| single       | answer_relevancy   |         -0.0627 | [-0.188, 0.060]  | False                  |         -0.0618 | [-0.187, 0.060]  | False                  |
| single       | hallucination_rate |          0.1481 | [-0.074, 0.370]  | False                  |          0.1481 | [-0.074, 0.370]  | False                  |
| single       | answer_correctness |         -0.1111 | [-0.259, 0.037]  | False                  |         -0.1111 | [-0.259, 0.037]  | False                  |
| multihop     | context_precision  |         -0.1857 | [-0.279, -0.100] | True                   |         -0.1857 | [-0.279, -0.100] | True                   |
| multihop     | faithfulness       |         -0.0325 | [-0.093, 0.033]  | False                  |         -0.0325 | [-0.093, 0.033]  | False                  |
| multihop     | answer_relevancy   |          0.0432 | [-0.060, 0.159]  | False                  |          0.0432 | [-0.060, 0.159]  | False                  |
| multihop     | hallucination_rate |          0.0571 | [-0.114, 0.229]  | False                  |          0.0571 | [-0.114, 0.229]  | False                  |
| multihop     | answer_correctness |          0.0286 | [-0.114, 0.171]  | False                  |          0.0286 | [-0.114, 0.171]  | False                  |
| unanswerable | answer_relevancy   |          0      | [0.000, 0.000]   | False                  |          0      | [0.000, 0.000]   | False                  |
| unanswerable | hallucination_rate |          0      | [0.000, 0.000]   | False                  |          0      | [0.000, 0.000]   | False                  |
| unanswerable | answer_correctness |          0      | [0.000, 0.000]   | False                  |          0      | [0.000, 0.000]   | False                  |

## 4. Danh sách judge failures và sensitivity analysis của retrieval precision

| system     | id   | step         | failure          |   missing |   expected | error                                                            |
|:-----------|:-----|:-------------|:-----------------|----------:|-----------:|:-----------------------------------------------------------------|
| vector_rag | q044 | passages     | json_error       |         4 |          4 |                                                                  |
| vector_rag | q092 | passages     | missing_verdicts |         1 |          4 |                                                                  |
| graphrag   | q015 | passages     | ollama_error     |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
| graphrag   | q061 | kg_relations | ollama_error     |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
| graphrag   | s002 | kg_relations | ollama_error     |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
| graphrag   | g026 | passages     | ollama_error     |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
| graphrag   | g043 | kg_relations | ollama_error     |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |

*failure: json_error (JSON hỏng) / ollama_error (Ollama huỷ lần sinh) / missing_verdicts (JSON hợp lệ nhưng thiếu verdict); verdict thiếu bị tính là "no". step passages: chấm 4 passage (retrieval precision); kg_relations: chấm relation của KG (metric phụ).*

| set                      | type     |   n |    diff |      lo |      hi | significant   |
|:-------------------------|:---------|----:|--------:|--------:|--------:|:--------------|
| all questions            | all      |  62 | -0.129  | -0.2056 | -0.0484 | True          |
| all questions            | single   |  27 | -0.0556 | -0.1852 |  0.0741 | False         |
| all questions            | multihop |  35 | -0.1857 | -0.2786 | -0.0929 | True          |
| excluding judge failures | all      |  58 | -0.1379 | -0.2112 | -0.0647 | True          |
| excluding judge failures | single   |  24 | -0.0833 | -0.2083 |  0.0417 | False         |
| excluding judge failures | multihop |  34 | -0.1765 | -0.2721 | -0.0882 | True          |

*Paired difference retrieval precision GraphRAG − Vector RAG, CI 95% bootstrap; "excluding judge failures": bỏ các câu có judge failure ở bước chấm passage của một trong hai hệ.*
