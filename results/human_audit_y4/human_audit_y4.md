# Chấm tay nội dung 4: context gốc (base) và nén cả hai (both)

30 câu hỏi × 2 chế độ; 177 ý ground truth, 165 ý bắt buộc (5.5 mỗi câu). Judge `llama3.1:8b`, điểm gộp 3 lần chấm.

| chỉ số                    |   base |   both | both − base                      |
|:--------------------------|-------:|-------:|:---------------------------------|
| faithfulness_human        | 1      | 0.987  | -0.013 [-0.033; +0.000] (n = 30) |
| faithfulness_judge        | 0.8602 | 0.8358 | -0.024 [-0.097; +0.039] (n = 30) |
| context_recall_human      | 0.8975 | 0.8007 | -0.097 [-0.150; -0.047] (n = 30) |
| context_recall_judge      | 0.7835 | 0.7307 | -0.053 [-0.102; -0.007] (n = 30) |
| context_recall_must_human | 0.903  | 0.8098 | -0.093 [-0.146; -0.046] (n = 30) |
| answer_recall_must_human  | 0.4972 | 0.4883 | -0.009 [-0.073; +0.048] (n = 30) |
| correct_must_human        | 0.2    | 0.2    | +0.000 [+0.000; +0.000] (n = 30) |
| correct_judge             | 0.9    | 0.9333 | +0.033 [+0.000; +0.100] (n = 30) |
| hallucination_human       | 0      | 0.0667 | +0.067 [+0.000; +0.167] (n = 30) |
| hallucination_judge       | 0.5333 | 0.5    | -0.033 [-0.167; +0.100] (n = 30) |

Độ lệch faithfulness judge − người: context gốc -0.140, context nén -0.151; chênh lệch nén − gốc -0.011 [-0.088; +0.058] (n = 30).

Khớp giữa "đúng theo ý bắt buộc" (người) và answer correctness (judge): 28.3%.
