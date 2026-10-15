# Kết quả ý 2: GraphRAG vs Vector RAG — `kg30_judge_audit`

30 bài · KG: 8352 entity, 9337 quan hệ · LLM: `qwen2.5:7b` · Judge: `llama3.1:8b` · Embedding: `nomic-embed-text` · TOP_K = 4 đoạn văn · version: xem `run_info.json`

## 1. Chỉ số chính

### Tất cả câu có đáp án (single-hop + multi-hop)

| system     |   n |   context_precision | context_precision_ci95   |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |   answer_correctness | answer_correctness_ci95   |
|:-----------|----:|--------------------:|:-------------------------|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|---------------------:|:--------------------------|
| vector_rag |  62 |              0.6169 | [0.548, 0.678]           |         0.8909 | [0.852, 0.928]      |             0.7426 | [0.679, 0.799]          |               0.4355 | [0.323, 0.565]            |               0.8548 | [0.774, 0.935]            |
| graphrag   |  62 |              0.4879 | [0.427, 0.544]           |         0.8468 | [0.808, 0.884]      |             0.7401 | [0.672, 0.797]          |               0.5323 | [0.403, 0.661]            |               0.8226 | [0.710, 0.919]            |

### Câu hỏi một đoạn (single-hop)

| system     |   n |   context_precision | context_precision_ci95   |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |   answer_correctness | answer_correctness_ci95   |
|:-----------|----:|--------------------:|:-------------------------|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|---------------------:|:--------------------------|
| vector_rag |  27 |              0.4907 | [0.389, 0.583]           |         0.9453 | [0.894, 0.985]      |             0.7654 | [0.666, 0.836]          |               0.1852 | [0.037, 0.333]            |               0.8519 | [0.704, 0.963]            |
| graphrag   |  27 |              0.4352 | [0.352, 0.528]           |         0.9036 | [0.847, 0.958]      |             0.7036 | [0.582, 0.803]          |               0.3333 | [0.185, 0.519]            |               0.7407 | [0.556, 0.889]            |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system     |   n |   context_precision | context_precision_ci95   |   faithfulness | faithfulness_ci95   |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |   answer_correctness | answer_correctness_ci95   |
|:-----------|----:|--------------------:|:-------------------------|---------------:|:--------------------|-------------------:|:------------------------|---------------------:|:--------------------------|---------------------:|:--------------------------|
| vector_rag |  35 |              0.7143 | [0.650, 0.779]           |         0.8469 | [0.797, 0.898]      |             0.7251 | [0.630, 0.801]          |               0.6286 | [0.457, 0.771]            |               0.8571 | [0.743, 0.971]            |
| graphrag   |  35 |              0.5286 | [0.443, 0.607]           |         0.8072 | [0.759, 0.857]      |             0.7682 | [0.698, 0.819]          |               0.6857 | [0.543, 0.829]            |               0.8857 | [0.771, 0.971]            |

### Câu hỏi ngoài corpus (đúng = từ chối; answer_relevancy của câu từ chối = 0)

| system     |   n |   answer_relevancy | answer_relevancy_ci95   |   hallucination_rate | hallucination_rate_ci95   |   answer_correctness | answer_correctness_ci95   |
|:-----------|----:|-------------------:|:------------------------|---------------------:|:--------------------------|---------------------:|:--------------------------|
| vector_rag |  23 |                  0 | [0.000, 0.000]          |                    0 | [0.000, 0.000]            |                    1 | [1.000, 1.000]            |
| graphrag   |  23 |                  0 | [0.000, 0.000]          |                    0 | [0.000, 0.000]            |                    1 | [1.000, 1.000]            |

*`_ci95`: khoảng tin cậy 95% (bootstrap 2000 lần trên các câu hỏi).*

## 2. Hiệu ghép cặp (graphrag − vector_rag)

| type         | metric             |   n_pairs |    diff | ci95             | significant   |
|:-------------|:-------------------|----------:|--------:|:-----------------|:--------------|
| all          | context_precision  |        62 | -0.129  | [-0.206, -0.048] | True          |
| all          | faithfulness       |        51 | -0.0303 | [-0.080, 0.020]  | False         |
| all          | answer_relevancy   |        62 | -0.0025 | [-0.090, 0.083]  | False         |
| all          | hallucination_rate |        62 |  0.0968 | [-0.048, 0.226]  | False         |
| all          | answer_correctness |        62 | -0.0323 | [-0.145, 0.081]  | False         |
| single       | context_precision  |        27 | -0.0556 | [-0.176, 0.074]  | False         |
| single       | faithfulness       |        22 | -0.0273 | [-0.098, 0.047]  | False         |
| single       | answer_relevancy   |        27 | -0.0618 | [-0.187, 0.060]  | False         |
| single       | hallucination_rate |        27 |  0.1481 | [-0.074, 0.370]  | False         |
| single       | answer_correctness |        27 | -0.1111 | [-0.259, 0.037]  | False         |
| multihop     | context_precision  |        35 | -0.1857 | [-0.279, -0.100] | True          |
| multihop     | faithfulness       |        29 | -0.0325 | [-0.093, 0.033]  | False         |
| multihop     | answer_relevancy   |        35 |  0.0432 | [-0.060, 0.159]  | False         |
| multihop     | hallucination_rate |        35 |  0.0571 | [-0.114, 0.229]  | False         |
| multihop     | answer_correctness |        35 |  0.0286 | [-0.114, 0.171]  | False         |
| unanswerable | answer_relevancy   |        23 |  0      | [0.000, 0.000]   | False         |
| unanswerable | hallucination_rate |        23 |  0      | [0.000, 0.000]   | False         |
| unanswerable | answer_correctness |        23 |  0      | [0.000, 0.000]   | False         |

- **context_precision** (retrieval precision, 0–1, cao = tốt): tỉ lệ đoạn văn (trong TOP_K đoạn mỗi hệ lấy ra)
  mà judge đánh giá là giúp suy ra đáp án chuẩn. Câu ngoài corpus không tính.
- **faithfulness / hallucination_rate**: như ý 1, chấm trên đúng context đưa vào LLM (GraphRAG: mô tả entity/
  quan hệ trong KG + đoạn văn). `*_text` (phụ): chỉ chấm trên đoạn văn.
- **answer_relevancy**: như ý 1 (câu từ chối = 0). **answer_correctness**: judge so với đáp án chuẩn
  (câu ngoài corpus: đúng = từ chối).
- **Hiệu ghép cặp**: graphrag − vector_rag trên cùng câu hỏi; `significant` = CI 95% không chứa 0.

## 3. Chỉ số phụ (chỉ để giải thích kết quả)

### Tất cả câu có đáp án (single-hop + multi-hop)

| system     |   context_ap |   source_precision |   retrieval_hit |   gold_chunk_recall |   context_recall |   kg_relation_precision |   faithfulness_text |   hallucination_rate_text |   refusal_rate |   latency_s |   llm_calls |
|:-----------|-------------:|-------------------:|----------------:|--------------------:|-----------------:|------------------------:|--------------------:|--------------------------:|---------------:|------------:|------------:|
| vector_rag |       0.7912 |             0.6855 |          0.7581 |              0.6855 |           0.8047 |                nan      |              0.8909 |                    0.4355 |         0.0968 |     10.8618 |           1 |
| graphrag   |       0.6241 |             0.5847 |          0.5161 |              0.3871 |           0.6917 |                  0.3212 |              0.7799 |                    0.5968 |         0.0968 |     15.185  |           1 |

### Câu hỏi một đoạn (single-hop)

| system     |   context_ap |   source_precision |   retrieval_hit |   gold_chunk_recall |   context_recall |   kg_relation_precision |   faithfulness_text |   hallucination_rate_text |   refusal_rate |   latency_s |   llm_calls |
|:-----------|-------------:|-------------------:|----------------:|--------------------:|-----------------:|------------------------:|--------------------:|--------------------------:|---------------:|------------:|------------:|
| vector_rag |       0.7078 |             0.6852 |          0.9259 |              0.7778 |           0.8284 |                nan      |              0.9453 |                    0.1852 |         0.0741 |     10.7696 |           1 |
| graphrag   |       0.6368 |             0.537  |          0.7778 |              0.4444 |           0.746  |                  0.2099 |              0.8975 |                    0.3333 |         0.1481 |     14.7881 |           1 |

### Câu hỏi tổng hợp 2 bài (multi-hop)

| system     |   context_ap |   source_precision |   retrieval_hit |   gold_chunk_recall |   context_recall |   kg_relation_precision |   faithfulness_text |   hallucination_rate_text |   refusal_rate |   latency_s |   llm_calls |
|:-----------|-------------:|-------------------:|----------------:|--------------------:|-----------------:|------------------------:|--------------------:|--------------------------:|---------------:|------------:|------------:|
| vector_rag |       0.8556 |             0.6857 |          0.6286 |              0.6143 |           0.7864 |                nan      |              0.8469 |                    0.6286 |         0.1143 |     10.9329 |           1 |
| graphrag   |       0.6143 |             0.6214 |          0.3143 |              0.3429 |           0.6498 |                  0.4071 |              0.6979 |                    0.8    |         0.0571 |     15.4911 |           1 |

### Câu hỏi ngoài corpus (đúng = từ chối; answer_relevancy của câu từ chối = 0)

| system     |   hallucination_rate_text |   refusal_rate |   latency_s |   llm_calls |
|:-----------|--------------------------:|---------------:|------------:|------------:|
| vector_rag |                         0 |              1 |      7.8883 |           1 |
| graphrag   |                         0 |              1 |     11.91   |           1 |

## Judge failures và truncate (log judge)

| system     | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   prompt_tokens_max |   prompt_tokens_median |
|:-----------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|--------------------:|-----------------------:|
| graphrag   | correctness       |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 361 |                    247 |
| graphrag   | extract_claims    |     118 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 248 |                    155 |
| graphrag   | reverse_questions |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 218 |                    139 |
| graphrag   | useful            |     124 |                5 |            0 |              5 |            0 |                        5 |                 44 |                  0 |                1545 |                    836 |
| graphrag   | verify_claims     |     174 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2163 |                   1573 |
| graphrag   | (tổng)            |     528 |                5 |            0 |              5 |            0 |                        5 |                 44 |                  0 |                2163 |                    566 |
| vector_rag | correctness       |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 416 |                    248 |
| vector_rag | extract_claims    |     118 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 251 |                    153 |
| vector_rag | reverse_questions |      56 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 236 |                    142 |
| vector_rag | useful            |      62 |                2 |            1 |              0 |            1 |                        2 |                  5 |                  0 |                1486 |                   1020 |
| vector_rag | verify_claims     |     118 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                1465 |                    999 |
| vector_rag | (tổng)            |     410 |                2 |            1 |              0 |            1 |                        2 |                  5 |                  0 |                1486 |                    256 |

*judge_failures: số lần gọi judge không trả đủ verdict hợp lệ = json_error (JSON hỏng) + ollama_error (Ollama huỷ lần sinh) + lần gọi thiếu verdict; verdicts_missing: số claim / đoạn văn không có verdict (bị tính là "không"); output_cut: đầu ra chạm num_predict; truncated_prompt: prompt dài hơn num_ctx = 4096 bị Ollama cắt. Chỉ gồm các câu được chấm sau khi có log.*
