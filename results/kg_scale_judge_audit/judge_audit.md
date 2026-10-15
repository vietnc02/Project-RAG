# Judge audit ý 3: re-judge GraphRAG với KG10 / KG20 (KG30: results/kg30_judge_audit)

Judge `llama3.1:8b`, num_ctx = 4096. Answer giữ nguyên, chỉ chấm lại.

## 1. Judge failures và truncate (theo prompt)

| system   | prompt            |   calls |   judge_failures |   json_error |   ollama_error |   output_cut |   calls_missing_verdicts |   verdicts_missing |   truncated_prompt |   prompt_tokens_max |   prompt_tokens_median | k   |
|:---------|:------------------|--------:|-----------------:|-------------:|---------------:|-------------:|-------------------------:|-------------------:|-------------------:|--------------------:|-----------------------:|:----|
| graphrag | correctness       |      39 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 379 |                    246 | k10 |
| graphrag | extract_claims    |     101 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 248 |                    155 | k10 |
| graphrag | reverse_questions |      39 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 199 |                    138 | k10 |
| graphrag | useful            |     124 |               12 |            0 |             12 |            0 |                       12 |                128 |                  0 |                1231 |                    908 | k10 |
| graphrag | verify_claims     |     140 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                1839 |                   1486 | k10 |
| graphrag | (tổng)            |     443 |               12 |            0 |             12 |            0 |                       12 |                128 |                  0 |                1839 |                    570 | k10 |
| graphrag | correctness       |      48 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 361 |                    249 | k20 |
| graphrag | extract_claims    |     110 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 248 |                    159 | k20 |
| graphrag | reverse_questions |      48 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                 182 |                    143 | k20 |
| graphrag | useful            |     124 |               11 |            0 |             11 |            0 |                       11 |                116 |                  0 |                1544 |                    897 | k20 |
| graphrag | verify_claims     |     158 |                0 |            0 |              0 |            0 |                        0 |                  0 |                  0 |                2247 |                   1538 | k20 |
| graphrag | (tổng)            |     488 |               11 |            0 |             11 |            0 |                       11 |                116 |                  0 |                2247 |                    581 | k20 |

- Tổng (KG10 + KG20): 931 judge call, judge_failures 23, truncated prompt 0, prompt dài nhất 2247 token.

## 2. Danh sách judge failures (KG10, KG20 và KG30)

|   k | id   | step         | failure      |   missing |   expected | error                                                            |
|----:|:-----|:-------------|:-------------|----------:|-----------:|:-----------------------------------------------------------------|
|  10 | q001 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | q006 | passages     | ollama_error |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | q006 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | s006 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | m003 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g005 | passages     | ollama_error |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g011 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g027 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g028 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g040 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g045 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  10 | g050 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | q001 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | q006 | passages     | ollama_error |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | q057 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | s002 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | s007 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | m003 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | g001 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | g011 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | g028 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | g040 | passages     | ollama_error |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
|  20 | g044 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  30 | q015 | passages     | ollama_error |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
|  30 | q061 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  30 | s002 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |
|  30 | g026 | passages     | ollama_error |         4 |          4 | prediction aborted, token repeat limit reached (status code: -1) |
|  30 | g043 | kg_relations | ollama_error |        12 |         12 | prediction aborted, token repeat limit reached (status code: -1) |

## 3. Reproducibility (điểm gốc so với re-judge)

|   k | metric             |   n |   mean_original |   mean_rejudged |   exact_agreement |   mean_abs_diff |   cohen_kappa |
|----:|:-------------------|----:|----------------:|----------------:|------------------:|----------------:|--------------:|
|  10 | context_precision  |  62 |          0.4516 |          0.4315 |            0.8871 |          0.0363 |       nan     |
|  10 | faithfulness       |  39 |          0.7238 |          0.7715 |            0.6923 |          0.0618 |       nan     |
|  10 | answer_relevancy   |  85 |          0.365  |          0.3656 |            0.8118 |          0.0023 |       nan     |
|  10 | hallucination      |  85 |          0.2941 |          0.2824 |            0.9647 |        nan      |         0.914 |
|  10 | answer_correctness |  85 |          0.6706 |          0.6588 |            0.9882 |        nan      |         0.974 |
|  10 | refusal            |  85 |          0.5412 |          0.5412 |            1      |        nan      |         1     |
|  20 | context_precision  |  62 |          0.4597 |          0.4597 |            0.871  |          0.0403 |       nan     |
|  20 | faithfulness       |  48 |          0.8166 |          0.8045 |            0.8542 |          0.0319 |       nan     |
|  20 | answer_relevancy   |  85 |          0.4607 |          0.4608 |            0.7647 |          0.003  |       nan     |
|  20 | hallucination      |  85 |          0.3882 |          0.4118 |            0.9529 |        nan      |         0.902 |
|  20 | answer_correctness |  85 |          0.8235 |          0.8235 |            1      |        nan      |         1     |
|  20 | refusal            |  85 |          0.4353 |          0.4353 |            1      |        nan      |         1     |

|   k |   claims_compared |   claims_identical |   context_precision_differ |   context_precision_differ_with_failure |   faithfulness_differ |   faithfulness_differ_with_failure |
|----:|------------------:|-------------------:|---------------------------:|----------------------------------------:|----------------------:|-----------------------------------:|
|  10 |                39 |                 20 |                          7 |                                       3 |                    12 |                                  2 |
|  20 |                48 |                 35 |                          8 |                                       2 |                     7 |                                  2 |

*claims_identical: số câu mà judge tách ra đúng cùng danh sách claim ở hai lần chấm; *_differ: số câu có điểm khác nhau, *_differ_with_failure: trong đó có judge failure khi re-judge. Cùng Ollama 0.35.1 / digest model ở hai lần chấm gốc của ý 2 và ý 3, nhưng chỉ ý 2 tái lập y hệt.*

## 4. Sensitivity analysis: paired difference của Y3 khi bỏ các câu có judge failure

| set   | compare     | metric             |   n_all |   diff_all |   lo_all |   hi_all | significant_all   |   n_excl |   diff_excl |   lo_excl |   hi_excl | significant_excl   |   diff_rejudged |   lo_rejudged |   hi_rejudged | significant_rejudged   |
|:------|:------------|:-------------------|--------:|-----------:|---------:|---------:|:------------------|---------:|------------:|----------:|----------:|:-------------------|----------------:|--------------:|--------------:|:-----------------------|
| S10   | KG10 − KG30 | context_precision  |      15 |    -0.0167 |  -0.15   |   0.1333 | False             |       12 |     -0.1042 |   -0.2083 |    0      | False              |         -0.05   |       -0.1833 |        0.0833 | False                  |
| S10   | KG10 − KG30 | faithfulness       |      13 |     0.0057 |  -0.098  |   0.1066 | False             |       11 |      0.0219 |   -0.103  |    0.1387 | False              |          0.082  |       -0.0438 |        0.2042 | False                  |
| S10   | KG10 − KG30 | answer_relevancy   |      15 |     0.0442 |  -0.0266 |   0.1624 | False             |       12 |     -0.0047 |   -0.0229 |    0.015  | False              |          0.0477 |       -0.0192 |        0.1638 | False                  |
| S10   | KG10 − KG30 | hallucination      |      15 |    -0.1333 |  -0.4667 |   0.2    | False             |       12 |     -0.25   |   -0.5833 |    0.0833 | False              |         -0.1333 |       -0.4667 |        0.2017 | False                  |
| S10   | KG10 − KG30 | answer_correctness |      15 |     0.0667 |  -0.1333 |   0.2667 | False             |       12 |      0      |   -0.25   |    0.25   | False              |          0.0667 |       -0.1333 |        0.2667 | False                  |
| S10   | KG20 − KG30 | context_precision  |      15 |    -0.1    |  -0.2167 |   0.0167 | False             |       13 |     -0.0769 |   -0.2115 |    0.0385 | False              |         -0.1    |       -0.2333 |        0.0333 | False                  |
| S10   | KG20 − KG30 | faithfulness       |      12 |     0.0309 |  -0.066  |   0.1278 | False             |       10 |      0.0204 |   -0.0962 |    0.1308 | False              |          0.042  |       -0.08   |        0.1748 | False                  |
| S10   | KG20 − KG30 | answer_relevancy   |      15 |    -0.0161 |  -0.1845 |   0.1452 | False             |       13 |     -0.011  |   -0.2057 |    0.1723 | False              |         -0.0182 |       -0.1859 |        0.1433 | False                  |
| S10   | KG20 − KG30 | hallucination      |      15 |     0      |  -0.2667 |   0.2667 | False             |       13 |      0      |   -0.3077 |    0.3077 | False              |          0      |       -0.3333 |        0.3333 | False                  |
| S10   | KG20 − KG30 | answer_correctness |      15 |     0.0667 |   0      |   0.2    | False             |       13 |      0.0769 |    0      |    0.2308 | False              |          0.0667 |        0      |        0.2    | False                  |
| S10   | KG10 − KG20 | context_precision  |      15 |     0.0833 |  -0.05   |   0.2333 | False             |       13 |      0.0577 |   -0.0769 |    0.2115 | False              |          0.05   |       -0.05   |        0.15   | False                  |
| S10   | KG10 − KG20 | faithfulness       |      13 |     0.0008 |  -0.1046 |   0.1016 | False             |       11 |     -0.0043 |   -0.1234 |    0.1126 | False              |          0.0668 |        0.0099 |        0.1331 | True                   |
| S10   | KG10 − KG20 | answer_relevancy   |      15 |     0.0603 |  -0.0065 |   0.1844 | False             |       13 |      0.0669 |   -0.0099 |    0.2082 | False              |          0.0658 |       -0.0032 |        0.1891 | False                  |
| S10   | KG10 − KG20 | hallucination      |      15 |    -0.1333 |  -0.4    |   0.1333 | False             |       13 |     -0.1538 |   -0.4615 |    0.1538 | False              |         -0.1333 |       -0.4    |        0.1333 | False                  |
| S10   | KG10 − KG20 | answer_correctness |      15 |     0      |  -0.2    |   0.2    | False             |       13 |      0      |   -0.2308 |    0.2308 | False              |          0      |       -0.2    |        0.2    | False                  |
| S20   | KG20 − KG30 | context_precision  |      37 |    -0.0203 |  -0.0878 |   0.0405 | False             |       29 |     -0.0086 |   -0.0776 |    0.0603 | False              |         -0.0338 |       -0.1014 |        0.0338 | False                  |
| S20   | KG20 − KG30 | faithfulness       |      32 |    -0.0066 |  -0.0641 |   0.0496 | False             |       24 |     -0.0053 |   -0.0683 |    0.0572 | False              |         -0.0031 |       -0.0648 |        0.0611 | False                  |
| S20   | KG20 − KG30 | answer_relevancy   |      37 |    -0.0025 |  -0.0704 |   0.0611 | False             |       29 |     -0.0011 |   -0.0882 |    0.0817 | False              |         -0.0038 |       -0.072  |        0.0592 | False                  |
| S20   | KG20 − KG30 | hallucination      |      37 |     0      |  -0.1622 |   0.1622 | False             |       29 |      0      |   -0.2069 |    0.1724 | False              |          0.027  |       -0.1622 |        0.1892 | False                  |
| S20   | KG20 − KG30 | answer_correctness |      37 |     0.027  |   0      |   0.0811 | False             |       29 |      0.0345 |    0      |    0.1034 | False              |          0.027  |        0      |        0.0811 | False                  |
| all   | KG10 − KG30 | context_precision  |      62 |    -0.0363 |  -0.1129 |   0.0363 | False             |       46 |     -0.0598 |   -0.1467 |    0.0217 | False              |         -0.0565 |       -0.129  |        0.0121 | False                  |
| all   | KG10 − KG30 | faithfulness       |      37 |    -0.0871 |  -0.1633 |  -0.0127 | True              |       26 |     -0.0822 |   -0.186  |    0.0118 | False              |         -0.0378 |       -0.1152 |        0.0368 | False                  |
| all   | KG10 − KG30 | answer_relevancy   |      62 |    -0.24   |  -0.3463 |  -0.1335 | True              |       46 |     -0.2552 |   -0.3787 |   -0.1365 | True               |         -0.2389 |       -0.3455 |       -0.1326 | True                   |
| all   | KG10 − KG30 | hallucination      |      62 |    -0.129  |  -0.2742 |   0.0161 | False             |       46 |     -0.1522 |   -0.3261 |    0.0217 | False              |         -0.1452 |       -0.2903 |        0      | False                  |
| all   | KG10 − KG30 | answer_correctness |      62 |    -0.2742 |  -0.4032 |  -0.1452 | True              |       46 |     -0.3478 |   -0.5    |   -0.1957 | True               |         -0.2903 |       -0.4194 |       -0.1613 | True                   |
| all   | KG20 − KG30 | context_precision  |      62 |    -0.0282 |  -0.0847 |   0.0242 | False             |       47 |     -0.0266 |   -0.0851 |    0.0319 | False              |         -0.0282 |       -0.0887 |        0.0323 | False                  |
| all   | KG20 − KG30 | faithfulness       |      47 |    -0.0324 |  -0.0848 |   0.0186 | False             |       34 |     -0.0318 |   -0.0925 |    0.0284 | False              |         -0.0447 |       -0.0972 |        0.01   | False                  |
| all   | KG20 − KG30 | answer_relevancy   |      62 |    -0.1088 |  -0.1926 |  -0.0341 | True              |       47 |     -0.1271 |   -0.2269 |   -0.0334 | True               |         -0.1084 |       -0.1916 |       -0.0336 | True                   |
| all   | KG20 − KG30 | hallucination      |      62 |     0      |  -0.1452 |   0.1452 | False             |       47 |     -0.0213 |   -0.1915 |    0.1489 | False              |          0.0323 |       -0.1129 |        0.1774 | False                  |
| all   | KG20 − KG30 | answer_correctness |      62 |    -0.0645 |  -0.1613 |   0.0161 | False             |       47 |     -0.0638 |   -0.1702 |    0.0426 | False              |         -0.0645 |       -0.1613 |        0.0161 | False                  |

*_all: điểm gốc, đủ câu; _excl: điểm gốc, bỏ các câu có judge failure ở một trong hai KG được so; _rejudged: điểm re-judge, đủ câu. Số so sánh đổi kết luận (significant) ở ít nhất một cách: 2.*

### Các so sánh đổi kết luận

| set   | compare     | metric       |   n_all |   diff_all |   lo_all |   hi_all | significant_all   |   n_excl |   diff_excl |   lo_excl |   hi_excl | significant_excl   |   diff_rejudged |   lo_rejudged |   hi_rejudged | significant_rejudged   |
|:------|:------------|:-------------|--------:|-----------:|---------:|---------:|:------------------|---------:|------------:|----------:|----------:|:-------------------|----------------:|--------------:|--------------:|:-----------------------|
| S10   | KG10 − KG20 | faithfulness |      13 |     0.0008 |  -0.1046 |   0.1016 | False             |       11 |     -0.0043 |   -0.1234 |    0.1126 | False              |          0.0668 |        0.0099 |        0.1331 | True                   |
| all   | KG10 − KG30 | faithfulness |      37 |    -0.0871 |  -0.1633 |  -0.0127 | True              |       26 |     -0.0822 |   -0.186  |    0.0118 | False              |         -0.0378 |       -0.1152 |        0.0368 | False                  |
