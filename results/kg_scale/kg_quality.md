# Chất lượng KG theo số bài (k = 10 / 20 / 30)

<!-- main -->
Bộ chuẩn: 62 câu có đáp án, 262 thực thể, 250 quan hệ (trích từ câu hỏi + đáp án chuẩn, duyệt tay).

| set              |   k |   n_questions |   n_gold_entities |   entity_coverage | entity_coverage_ci95   |   n_gold_relations |   relation_completeness | relation_completeness_ci95   |
|:-----------------|----:|--------------:|------------------:|------------------:|:-----------------------|-------------------:|------------------------:|:-----------------------------|
| trong phạm vi KG |  10 |            15 |                74 |            0.9054 | [0.838, 0.959]         |                 59 |                  0.3898 | [0.271, 0.508]               |
| trong phạm vi KG |  20 |            37 |               167 |            0.9042 | [0.862, 0.946]         |                151 |                  0.3709 | [0.291, 0.450]               |
| trong phạm vi KG |  30 |            62 |               262 |            0.8931 | [0.855, 0.927]         |                250 |                  0.416  | [0.356, 0.476]               |
| S10 (10 bài đầu) |  10 |            15 |                74 |            0.9054 | [0.838, 0.959]         |                 59 |                  0.3898 | [0.271, 0.508]               |
| S10 (10 bài đầu) |  20 |            15 |                74 |            0.9324 | [0.865, 0.986]         |                 59 |                  0.4068 | [0.288, 0.525]               |
| S10 (10 bài đầu) |  30 |            15 |                74 |            0.9324 | [0.865, 0.986]         |                 59 |                  0.4407 | [0.322, 0.559]               |
| S20 (20 bài đầu) |  20 |            37 |               167 |            0.9042 | [0.862, 0.946]         |                151 |                  0.3709 | [0.291, 0.450]               |
| S20 (20 bài đầu) |  30 |            37 |               167 |            0.9162 | [0.874, 0.958]         |                151 |                  0.4106 | [0.331, 0.490]               |
| toàn bộ 62 câu   |  10 |            62 |               262 |            0.6412 | [0.580, 0.698]         |                250 |                  0.204  | [0.156, 0.256]               |
| toàn bộ 62 câu   |  20 |            62 |               262 |            0.7748 | [0.721, 0.824]         |                250 |                  0.308  | [0.252, 0.364]               |
| toàn bộ 62 câu   |  30 |            62 |               262 |            0.8931 | [0.855, 0.927]         |                250 |                  0.416  | [0.356, 0.476]               |

Cấu trúc KG:

|   k |   papers |   chunks |   entities |   relations |   entities_per_paper |   entities_2plus_share |   relations_2plus_share |   avg_degree |   largest_component_share |   isolated_share |
|----:|---------:|---------:|-----------:|------------:|---------------------:|-----------------------:|------------------------:|-------------:|--------------------------:|-----------------:|
|  10 |       10 |      854 |       3274 |        3252 |                327.4 |                 0.0806 |                  0.0077 |        1.987 |                    0.6478 |           0.2071 |
|  20 |       20 |     1646 |       5702 |        6160 |                285.1 |                 0.1119 |                  0.0185 |        2.161 |                    0.657  |           0.2148 |
|  30 |       30 |     2528 |       8352 |        9337 |                278.4 |                 0.125  |                  0.024  |        2.236 |                    0.6616 |           0.2205 |

<!-- main -->

## Chỉ số phụ

| set      |   k |   entity_coverage_lenient |   relation_completeness_lenient |   relation_completeness_2hop |   relation_endpoints_covered |   questions_all_entities_covered |   gold_chunk_reachable |
|:---------|----:|--------------------------:|--------------------------------:|-----------------------------:|-----------------------------:|---------------------------------:|-----------------------:|
| in_scope |  10 |                    0.9459 |                          0.5593 |                       0.5593 |                       0.8475 |                           0.5333 |                 1      |
| in_scope |  20 |                    0.9461 |                          0.5695 |                       0.6291 |                       0.8808 |                           0.6216 |                 0.9595 |
| in_scope |  30 |                    0.9275 |                          0.56   |                       0.652  |                       0.84   |                           0.6129 |                 0.9677 |
| S10      |  10 |                    0.9459 |                          0.5593 |                       0.5593 |                       0.8475 |                           0.5333 |                 1      |
| S10      |  20 |                    0.9595 |                          0.6102 |                       0.6102 |                       0.8814 |                           0.6667 |                 1      |
| S10      |  30 |                    0.9595 |                          0.6271 |                       0.6949 |                       0.8814 |                           0.6667 |                 1      |
| S20      |  20 |                    0.9461 |                          0.5695 |                       0.6291 |                       0.8808 |                           0.6216 |                 0.9595 |
| S20      |  30 |                    0.9521 |                          0.6093 |                       0.6623 |                       0.894  |                           0.6486 |                 0.9595 |
| all      |  10 |                    0.7099 |                          0.288  |                       0.344  |                       0.548  |                           0.2903 |                 0.3629 |
| all      |  20 |                    0.8244 |                          0.44   |                       0.508  |                       0.716  |                           0.4839 |                 0.6694 |
| all      |  30 |                    0.9275 |                          0.56   |                       0.652  |                       0.84   |                           0.6129 |                 0.9677 |

Kiểm tra tay 40 cạnh KG30 khớp với quan hệ chuẩn (`kg_relation_audit.csv`): 36/40 cạnh nói đúng quan hệ chuẩn (90%).

## Đường cong tăng trưởng (`kg_growth.csv`)

Số entity mới trung bình mỗi bài thêm vào (20 thứ tự bài ngẫu nhiên): `{"2-10": 300.6, "11-20": 280.2, "21-30": 250.5}`

Kiểm tra phép chiếu KG30 -> k bài (dùng cho đường cong) so với KG dựng thật: `{"10": {"entities_projected": 3278, "entities_built": 3274, "relations_projected": 3261, "relations_built": 3252}, "20": {"entities_projected": 5705, "entities_built": 5702, "relations_projected": 6165, "relations_built": 6160}, "30": {"entities_projected": 8352, "entities_built": 8352, "relations_projected": 9337, "relations_built": 9337}}`

## Định nghĩa

- **entity coverage**: tỉ lệ thực thể chuẩn có trong KG (trùng tên sau chuẩn hoá, cùng quy tắc gộp entity của `kg_build`). `_lenient` (cận trên): thêm entity có tên chứa trọn cụm từ của thực thể chuẩn.
- **relation completeness**: tỉ lệ quan hệ chuẩn có cạnh trực tiếp trong KG giữa 2 thực thể đã khớp; `_lenient`: đầu mút khớp lenient; `_2hop`: nối nhau trong <= 2 bước; `relation_endpoints_covered`: cả 2 đầu mút có trong KG.
- **gold_chunk_reachable**: tỉ lệ đoạn gốc của câu hỏi có ít nhất 1 entity/quan hệ trong KG (GraphRAG chỉ lấy được đoạn văn qua entity/quan hệ).
- `*_2plus_share`: tỉ lệ entity / quan hệ xuất hiện ở >= 2 bài (liên kết giữa các bài).