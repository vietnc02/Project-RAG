# Đồ án RAG trên bộ bài báo khoa học (server local)

- **Ý 1** — LangChain RAG thuần vs LangGraph Self-RAG: faithfulness, answer relevancy, hallucination rate
  (báo cáo `report/Bao_cao_Y1.pdf`)
- **Ý 2** — GraphRAG (Knowledge Graph từ 30 bài chủ đề quantum) vs Vector RAG thuần: retrieval precision + generation quality
  (báo cáo `report/Bao_cao_Y2.pdf`, xem [phần Ý 2](#ý-2--graphrag-vs-vector-rag-30-bài-chủ-đề-quantum))
- **Ý 3** — KG từ k = 10, 20, 30 bài: chất lượng KG (entity coverage, relation completeness) + chất lượng câu trả lời,
  k bài có đủ để KG hoạt động tốt không (báo cáo `report/Bao_cao_Y3.pdf`, xem [phần Ý 3](#ý-3--kg-từ-k--10-20-30-bài))

## Chạy nhanh (sau khi clone repo)
Repo đã có sẵn dữ liệu (chunks + Chroma + Knowledge Graph), không cần build lại.
1. Cài [Ollama](https://ollama.com) 0.35.x và tải 3 model:
   ```powershell
   ollama pull qwen2.5:7b; ollama pull llama3.1:8b; ollama pull nomic-embed-text
   ```
2. Cài thư viện (Python 3.14): `pip install -r requirements.txt`
3. Chạy:
   ```powershell
   python rag_langchain.py "What is a network tarpit?"   # hệ A
   python rag_selfrag.py   "What is a network tarpit?"   # hệ B (in ra trace)
   python evaluate.py                                    # chạy lại toàn bộ đánh giá ý 1
   python rag_graphrag.py  "Which signature schemes did NIST select for standardization?"   # GraphRAG (ý 2)
   python evaluate_graphrag.py --run kg30                # chạy lại đánh giá ý 2
   python kg_quality.py --all                            # chất lượng KG10 / KG20 / KG30 (ý 3, không cần LLM)
   python evaluate_kg_scale.py                           # chạy lại đánh giá câu trả lời ý 3
   ```
   Lần chạy đầu, `data/chroma.zip` tự giải nén ra `data/chroma/` (có kiểm tra sha256).

## Môi trường
| Thành phần | Version |
|---|---|
| Python | 3.14.6 |
| langchain / langchain-core | 1.4.3 / 1.6.6 |
| langgraph | 1.2.12 |
| langchain-ollama / langchain-chroma | 1.1.0 / 1.1.0 |
| chromadb | 1.5.9 |
| Ollama server | 0.35.0; phần cuối run `k8` chạy trên 0.35.1 (Ollama tự cập nhật, digest model không đổi). `http://localhost:11434` |
| LLM sinh câu trả lời | `qwen2.5:7b` Q4_K_M (digest `845dbda0ea48`) |
| Judge | `llama3.1:8b` Q4_K_M (digest `46e0c10c039e`) |
| Embedding | `nomic-embed-text` F16 (digest `0a109f422b47`) |
| Phần cứng | i3-12100F, RAM 16 GB, GTX 1660 Ti 6 GB |

Toàn bộ thư viện Python đã pin trong `requirements.txt`. Mỗi lần chạy `evaluate.py` cũng ghi lại version vào thư mục kết quả.
Trên máy tác giả: venv `D:\RAG_env\venv` (`D:\RAG_env\venv\Scripts\Activate.ps1`), model Ollama ở `D:\ollama_models`.

## Dữ liệu (chunks + Chroma, có trong repo)
| File | Nội dung |
|---|---|
| `data/chunks.jsonl` | 6.649 chunk (văn bản + metadata nguồn/trang) từ 95 bài báo, chunk 1000 / overlap 150 |
| `data/chroma.zip` | Vector DB Chroma đúng như lúc chạy thí nghiệm (collection `papers`, cosine, 6.649 vector) |
| `data/MANIFEST.json` | Số lượng, sha256 của 2 file trên, tham số chunking, version thư viện / Ollama / model |

`data/chroma/` (bản giải nén) không commit vì `chroma.sqlite3` bị ghi lại mỗi lần mở.
```powershell
python ingest.py --restore       # giải nén lại chroma.zip (thường tự động)
python ingest.py --from-chunks   # embed lại từ data/chunks.jsonl
python ingest.py --reset         # build lại từ đầu từ PDF trong papers/
python ingest.py --snapshot      # (sau khi build lại) cập nhật chroma.zip + MANIFEST.json
```

# Ý 1 — LangChain RAG thuần vs LangGraph Self-RAG

## Quy trình
```powershell
python ingest.py --reset                 # 1. (tuỳ chọn) build lại Chroma từ PDF
python rag_langchain.py "câu hỏi"        #    thử hệ thống A
python rag_selfrag.py   "câu hỏi"        #    thử hệ thống B (in ra trace)
python generate_testset.py               # 2a. sinh câu single-hop -> testset/testset.json
python generate_testset.py --multihop 60 # 2b. sinh câu multi-hop  -> testset/multihop_raw.json
python generate_testset.py --multihop 40 --append --seed 7   #  sinh thêm, không lặp cặp bài đã có
#   -> DUYỆT TAY từng câu (đối chiếu đoạn gốc), kết quả duyệt ghi trong:
python testset/review_edits.py           #    single-hop -> testset/testset.json
python testset/review_multihop.py        #    multi-hop  -> testset/multihop.json
#   -> câu ngoài corpus viết tay: testset/unanswerable.json (có kiểm tra không xuất hiện trong chunks)
python evaluate.py --run k4                                                    # 3a. top-k = 4
python evaluate.py --run k8 --top-k 8 --testset testset/testset.json testset/multihop.json  # 3b. retrieval nhiễu
```
Judge mặc định là `llama3.1:8b` (khác họ với LLM sinh câu trả lời để tránh tự chấm bài mình).
Đổi judge: `$env:JUDGE_MODEL='<model>'; python evaluate.py --run <tên mới>`.

## Hai hệ thống ý 1 (dùng CHUNG retriever, top-k, prompt sinh câu trả lời, LLM)
- **A. LangChain RAG** (`rag_langchain.py`): retrieve top-k → generate. 1 lần gọi LLM.
- **B. LangGraph Self-RAG** (`rag_selfrag.py`): retrieve → chấm độ liên quan từng tài liệu →
  generate → tự kiểm tra *grounded* (hallucination, prompt có few-shot) → tự kiểm tra *useful* →
  nếu sai thì sinh lại (prompt chặt hơn) / viết lại truy vấn / từ chối trả lời.

## Bộ câu hỏi ý 1
| File | Loại | Số câu | Mục đích |
|---|---|---|---|
| `testset/testset.json` | single-hop: hỏi sự kiện trong 1 đoạn | 86 (sinh 93, duyệt tay) | trường hợp cơ bản |
| `testset/multihop.json` | multi-hop: cần tổng hợp 2 bài báo | 35 (sinh 71, duyệt tay) | câu hỏi khó, cần truy xuất nhiều nguồn |
| `testset/unanswerable.json` | ngoài corpus | 23 (viết tay, đã kiểm tra không có trong chunks) | hệ thống phải biết từ chối (abstain) |
| (`--top-k 8`) | retrieval nhiễu | single + multi-hop | context chứa nhiều đoạn không liên quan |

## Ba chỉ số so sánh (`evaluate.py`, định nghĩa theo RAGAS)
| Chỉ số | Định nghĩa |
|---|---|
| faithfulness ↑ | #claim được context hỗ trợ / #claim |
| answer_relevancy ↑ | cosine(câu hỏi, 3 câu hỏi sinh ngược từ câu trả lời); 0 nếu né tránh / từ chối |
| hallucination_rate ↓ | % câu hỏi có câu trả lời chứa ≥ 1 claim không được context hỗ trợ |

Faithfulness / hallucination của **cả 2 hệ** được chấm trên cùng một context: full top-k ban đầu của câu hỏi
gốc (Self-RAG lọc bớt tài liệu trước khi sinh, nếu chấm trên tài liệu đã lọc thì không công bằng).
Chỉ số phụ chỉ để giải thích: `*_used` (chấm trên tài liệu đưa vào generate), refusal_rate, answer_correctness,
retrieval_hit, claim_halluc_rate, latency_s, llm_calls.

Kết quả: `results/<run>/summary.md`, `summary.csv` (kèm CI 95% bootstrap), `per_question.csv`, `run_info.json`.

# Ý 2 — GraphRAG vs Vector RAG (30 bài chủ đề quantum)

Dựng Knowledge Graph (KG) từ 30 bài báo bằng LLM local, so sánh **GraphRAG** (truy xuất qua KG) với **Vector RAG**
(LangChain RAG của ý 1, embedding chunk) trên cùng 30 bài. Hai hệ dùng chung chunk, chung 4 đoạn văn mỗi câu, chung
prompt + LLM: khác biệt chỉ ở cách truy xuất. Báo cáo: `report/Bao_cao_Y2.pdf`.

**30 bài = một chủ đề chính (quantum)**, gồm 2 nhánh gần nhau để các bài có thực thể / khái niệm chung tạo quan hệ:
23 bài Quantum Security + Quantum Machine Learning có trong `papers/` và 7 bài bổ sung (truy cập mở trên arXiv, bài gốc
của các khái niệm mà 23 bài kia nhắc nhiều nhất: Shor, BB84, decoy-state QKD, RSA-2048, barren plateau, VQA, NISQ),
xem `papers_y2/SOURCES.md`. **Dữ liệu ý 1 không đổi**: 7 bài nằm trong `papers_y2/` (ngoài `papers/`), chunk của chúng
trong `data/kg/chunks_y2.jsonl`, và Vector RAG của ý 2 dùng Chroma riêng `data/kg/chroma.zip` (collection `kg30`).

## Quy trình
```powershell
python kg_build.py --ingest              # 1. chia chunk 7 bài trong papers_y2/ -> data/kg/chunks_y2.jsonl
python kg_build.py --select              # 2. chọn 30 bài -> data/kg/papers.json
python kg_build.py --store               # 3. Chroma riêng của ý 2 -> data/kg/chroma.zip + MANIFEST.json
python kg_build.py --extract --k 30      # 4. LLM trích entity/quan hệ từng chunk -> data/kg/extractions.jsonl
python kg_build.py --build --k 30        # 5. gộp entity + embed -> data/kg/k30/graph.json, vectors.npz
python rag_graphrag.py "câu hỏi"         #    thử GraphRAG
python generate_testset.py --kg                          # 6a. single-hop cho bài chưa có câu -> testset/singlehop_kg_raw.json
python generate_testset.py --multihop 30 --kg            # 6b. multi-hop trong 30 bài (+ --append --seed 7/11/13/17/19)
python testset/review_singlehop_kg.py                    #     kết quả duyệt tay -> testset/singlehop_kg.json
python testset/review_multihop_kg.py                     #     kết quả duyệt tay -> testset/multihop_kg.json
python evaluate_graphrag.py --run kg30                   # 7. đánh giá -> results/kg30/
python report/make_figures_y2.py; python report/build_report_y2.py   # 8. báo cáo -> report/Bao_cao_Y2.docx/.pdf
```
Các bước 1–6 đã chạy sẵn, kết quả có trong repo. `--k 10` / `--k 20` dựng KG trên 10 / 20 bài đầu (ý 3), không phải
trích xuất lại.

## Dữ liệu ý 2 (có trong repo)
| File | Nội dung |
|---|---|
| `papers_y2/` | 7 PDF bổ sung + `SOURCES.md` (nguồn arXiv, lý do chọn) |
| `data/kg/chunks_y2.jsonl` | chunk của 7 bài bổ sung (cùng cách chia chunk với ý 1) |
| `data/kg/chroma.zip` | Chroma của ý 2: chunk của đúng 30 bài (23 bài chép nguyên vector từ ý 1 + 7 bài mới); tự giải nén ra `data/kg/chroma/`, kiểm tra sha256 với `data/kg/MANIFEST.json` |
| `data/kg/papers.json` | 30 bài (Quantum Security 21, Quantum ML 9), thứ tự phân tầng theo chủ đề để k bài đầu giữ tỉ lệ chủ đề |
| `data/kg/extractions.jsonl` | entity + quan hệ do qwen2.5:7b trích từ từng chunk |
| `data/kg/k30/graph.json`, `vectors.npz` | KG đã gộp (mỗi entity/quan hệ giữ chunk nguồn) + vector entity/quan hệ |

Bước trích xuất đặt toàn bộ LLM lên GPU (`KG_NUM_GPU = 99` trong `config.py`): mặc định Ollama chỉ đưa ~82% model
lên GPU 6 GB và chạy phần còn lại trên CPU (chậm hơn 1,5 lần). Máy ít VRAM hơn: `$env:KG_NUM_GPU='-1'`.

## Hai hệ thống ý 2
- **Vector RAG** (`rag_langchain.py`, Chroma của ý 2 qua `kg_build.use_kg_store()` trong `evaluate_graphrag.py`):
  lấy 4 chunk gần câu hỏi nhất theo cosine.
- **GraphRAG** (`rag_graphrag.py`, local search, không gọi thêm LLM): khớp câu hỏi với entity (cosine + tên xuất hiện
  nguyên văn) → 8 entity hạt giống → mở rộng 1 bước, lấy 12 quan hệ → chọn 4 chunk có nhiều entity/quan hệ đã chọn.
  Context = mô tả entity + quan hệ + 4 đoạn văn.

## Bộ câu hỏi ý 2 (85 câu, mọi câu có đáp án đều thuộc 30 bài)
| Loại | Số câu | Nguồn |
|---|---|---|
| single-hop | 27 | 21 câu của ý 1 + 6 câu mới cho bài chưa có câu (`testset/singlehop_kg.json`: sinh 7, duyệt tay) |
| multi-hop | 35 | 9 câu của ý 1 + 26 câu mới trong 30 bài (`testset/multihop_kg.json`, duyệt tay) |
| ngoài corpus | 23 | câu của ý 1 (đã kiểm tra lại từ khoá đáp án không có trong 7 bài bổ sung) |

## Chỉ số và kết quả (`results/kg30/summary.md`)
Retrieval precision = context precision (RAGAS): tỉ lệ đoạn (trong 4 đoạn) giúp suy ra đáp án chuẩn. Generation
quality: faithfulness, answer relevancy, hallucination rate (như ý 1, chấm trên đúng context đưa vào LLM) + answer
correctness.

| 62 câu có đáp án | Vector RAG | GraphRAG | GraphRAG − Vector (CI 95%) |
|---|---|---|---|
| Retrieval precision ↑ | 0,62 | 0,49 | −0,13 [−0,21; −0,05] (có ý nghĩa) |
| Faithfulness ↑ | 0,89 | 0,85 | −0,03 (không ý nghĩa) |
| Answer relevancy ↑ | 0,74 | 0,74 | −0,00 (không ý nghĩa) |
| Hallucination rate ↓ | 44% | 53% | +10 điểm % (không ý nghĩa) |
| Answer correctness ↑ | 85% | 82% | −3 điểm % (không ý nghĩa) |

GraphRAG truy xuất kém hơn chủ yếu ở câu multi-hop (retrieval precision 0,53 so với 0,71); generation quality không khác
biệt có ý nghĩa (ở câu multi-hop answer correctness 89% so với 86%); câu ngoài corpus cả hai hệ từ chối đúng 23/23.
Phân tích nguyên nhân và đối chiếu tài liệu (Han et al. 2025, GraphRAG-Bench, HybridRAG) trong báo cáo.

# Ý 3 — KG từ k = 10, 20, 30 bài

Dựng KG từ **10, 20 và 30 bài đầu** của `data/kg/papers.json` (30 bài chủ đề quantum của ý 2, thứ tự phân tầng theo
chủ đề cố định từ ý 2), đo **chất lượng KG** (entity coverage, relation completeness) và **chất lượng câu trả lời** của
GraphRAG (ý 2) trên từng KG. Ba KG dùng chung cache trích xuất của ý 2 nên **lồng nhau** (KG10 ⊂ KG20 ⊂ KG30): khác
biệt chỉ đến từ số bài. Báo cáo: `report/Bao_cao_Y3.pdf`. Ý 3 chỉ thêm file mới, không sửa mã / dữ liệu của ý 1, ý 2.

## Quy trình
```powershell
python kg_build.py --build --k 10        # 1. KG10 -> data/kg/k10/ (gộp lại từ data/kg/extractions.jsonl, không trích lại)
python kg_build.py --build --k 20        #    KG20 -> data/kg/k20/   (KG30 = data/kg/k30/ của ý 2)
python kg_quality.py --gold              # 2. judge trích thực thể / quan hệ chuẩn của 62 câu -> testset/kg_gold_raw.json
python testset/review_kg_gold.py         #    kết quả duyệt tay -> testset/kg_gold.json
python kg_quality.py --all               # 3. chất lượng KG -> results/kg_scale/kg_quality.md (tất định, không gọi LLM)
python evaluate_kg_scale.py              # 4. GraphRAG với KG10/20/30 trên 85 câu của ý 2 -> results/kg_scale/summary.md
python report/make_figures_y3.py; python report/build_report_y3.py   # 5. báo cáo -> report/Bao_cao_Y3.docx/.pdf
```
Các bước 1–4 đã chạy sẵn, kết quả có trong repo. Kết quả GraphRAG với KG30 lấy lại từ `results/kg30/` của ý 2 (cùng mã,
KG, câu hỏi, môi trường).

## Dữ liệu ý 3 (có trong repo)
| File | Nội dung |
|---|---|
| `data/kg/k10/`, `data/kg/k20/` | KG10 (3.274 entity, 3.252 quan hệ), KG20 (5.702 / 6.160): `graph.json` + `vectors.npz` |
| `testset/kg_gold.json` | bộ chuẩn: 62 câu, 353 thực thể, 263 quan hệ (bản tự động `kg_gold_raw.json`: 503 / 415; duyệt tay trong `review_kg_gold.py`) |
| `results/kg_scale/` | chất lượng KG (`kg_quality.md`, `kg_coverage*.csv`, `kg_growth.csv`, `kg_relation_audit.csv`) + câu trả lời theo k (`k10/`, `k20/`, `k30/`, `summary.md`) |

## Cách đo
- **Entity coverage**: tỉ lệ thực thể chuẩn có trong KG, khớp theo tên sau chuẩn hoá (cùng quy tắc gộp entity của
  `kg_build.py`); bản `_lenient` (cận trên) tính thêm entity có tên chứa trọn cụm từ chuẩn. Không dùng LLM để khớp
  (đã thử llama3.1:8b, qwen2.5:7b: nhận nhầm khái niệm rộng / hẹp hơn).
- **Relation completeness**: tỉ lệ quan hệ chuẩn có cạnh trực tiếp trong KG nối 2 thực thể đã khớp; kiểm tra tay 40
  cạnh KG30: 36 cạnh nói đúng quan hệ chuẩn.
- **Câu trả lời**: 5 chỉ số của ý 2, xem theo 3 tập: *trong phạm vi KG* (bài nguồn thuộc k bài: 15 / 37 / 62 câu),
  *tập cố định* S10 (15 câu về 10 bài đầu) và S20 (37 câu về 20 bài đầu) so sánh ghép cặp, *toàn bộ 62 câu*.

## Kết quả
| | KG10 | KG20 | KG30 |
|---|---|---|---|
| Entity coverage (câu trong phạm vi KG) | 91% | 90% | 89% |
| Relation completeness (câu trong phạm vi KG) | 39% | 37% | 42% |
| Entity coverage / relation completeness (toàn bộ 62 câu) | 64% / 20% | 77% / 31% | 89% / 42% |
| Answer correctness (câu trong phạm vi KG) | 87% | 86% | 82% |
| Answer correctness (toàn bộ 62 câu) | 55% | 76% | 82% |
| Câu ngoài corpus từ chối đúng | 23/23 | 23/23 | 23/23 |

- Với câu hỏi về chính các bài trong KG, **10 bài đã đủ**: trên tập cố định S10 / S20 không chênh lệch nào giữa các k
  có ý nghĩa thống kê.
- Điểm yếu chung là **quan hệ** (~40% quan hệ cần thiết có trong KG), do mô hình trích xuất 7B chứ không do số bài.
- KG **chưa bão hoà**: mỗi bài thêm vào vẫn mang ~250–300 entity mới; trên toàn bộ 62 câu KG10 kém KG30 có ý nghĩa
  (−27 điểm %), KG20 thì không (−6,5 điểm %). Câu ngoài phạm vi KG vẫn được trả lời đúng 45% (KG10) / 60% (KG20), chủ
  yếu nhờ kiến thức sẵn có của mô hình (claim không có trong context).

Đối chiếu tài liệu trong báo cáo: Han et al. 2025, GraphRAG-Bench (thử nghiệm theo cỡ corpus), KGGen / MINE,
Zhu et al., Joren et al. (sufficient context, ICLR 2025), định luật Heaps.
