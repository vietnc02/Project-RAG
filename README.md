# Đồ án RAG trên bộ bài báo khoa học (server local)

So sánh các kiến trúc RAG (RAG thuần, Self-RAG, GraphRAG) trên 95 bài báo khoa học, chạy hoàn toàn cục bộ bằng Ollama
(LLM `qwen2.5:7b`, judge `llama3.1:8b`, embedding `nomic-embed-text`). Mỗi nội dung có báo cáo PDF riêng trong `report/`.

## Tóm tắt 4 nội dung

| | Câu hỏi | Kết quả chính | Báo cáo |
|---|---|---|---|
| **Ý 1** | LangChain RAG thuần vs LangGraph Self-RAG: faithfulness, answer relevancy, hallucination rate | Single-hop: không đủ bằng chứng để phân biệt hai hệ. Multi-hop: Self-RAG (xấp xỉ bằng prompt) giảm hallucination rate 71% → 37%, nhưng chủ yếu do refuse nhiều hơn (chỉ tính câu có trả lời: 72% so với 83%; số câu đúng, có căn cứ không tăng); đổi lại answer relevancy 0,70 → 0,42 | [Bao_cao_Y1.pdf](report/Bao_cao_Y1.pdf) · [chi tiết](#ý-1--langchain-rag-thuần-vs-langgraph-self-rag) |
| **Ý 2** | KG từ 30 bài (chủ đề quantum); GraphRAG vs Vector RAG: retrieval precision, generation quality | GraphRAG truy xuất kém hơn (retrieval precision 0,50 so với 0,63, statistically significant sau hiệu chỉnh Holm; gold chunk 44% so với 78% ở single-hop); generation quality không đủ bằng chứng để phân biệt. Agreement của judge với human judge thấp (κ 0,13–0,28) | [Bao_cao_Y2.pdf](report/Bao_cao_Y2.pdf) · [chi tiết](#ý-2--graphrag-vs-vector-rag-30-bài-chủ-đề-quantum) |
| **Ý 3** | KG từ k = 10, 20, 30 bài: k bài có đủ để KG hoạt động tốt không | Câu hỏi về chính các bài trong KG: không phát hiện khác biệt trên S10 / S20; năng lực kiểm định thấp. Điểm yếu là relation (~37% sau hiệu chỉnh). Hỏi cả chủ đề: answer correctness 53% / 76% / 82%. Theo tiêu chí non-inferiority δ = 10 điểm %: KG20 đủ cho câu về 20 bài đầu; KG10 chưa kết luận được cho câu về 10 bài đầu; với cả 30 bài, KG10 không đủ, KG20 chưa kết luận được. So với Vector RAG chỉ truy xuất trong cùng k bài: answer correctness không đủ bằng chứng để phân biệt ở mọi k | [Bao_cao_Y3.pdf](report/Bao_cao_Y3.pdf) · [chi tiết](#ý-3--kg-từ-k--10-20-30-bài) |
| **Ý 4** | Context compression (nén KG và passage): token reduction 19–53%, giữ nguyên faithfulness | Token reduction 22,5% / 25,3% / 47,9%; chưa chứng minh được faithfulness giữ nguyên (non-inferiority, ngưỡng 0,05; chưa loại trừ được mức giảm tới 0,070–0,120); nén KG làm refuse nhiều hơn (chưa qua hiệu chỉnh đa so sánh); chọn câu theo độ liên quan giữ context recall tốt hơn xoá ngẫu nhiên / cắt đầu ở phần passage | [Bao_cao_Y4.pdf](report/Bao_cao_Y4.pdf) · [chi tiết](#ý-4--context-compression-cho-graphrag) |

Mọi chênh lệch được báo kèm CI 95% (bootstrap, paired trên cùng câu hỏi; seed cố định cho từng lần bootstrap, hàm
chung trong `report/stat_utils.py`). Mỗi bảng paired difference là một họ so sánh, p được hiệu chỉnh theo Holm; chỉ
chênh lệch qua hiệu chỉnh mới được gọi là có statistical significance. Hallucination được tách khỏi refusal bằng nhóm
kết cục (đúng có căn cứ / đúng nhưng có claim không được hỗ trợ / sai / refuse). Mọi bảng của ý 2–4 và bảng chính của
ý 1 (top-k = 4) do judge chấm 3 lần (cùng cách chấm, temperature 0; nhãn nhị phân lấy đa số, điểm liên tục lấy trung
bình); mọi lần gọi judge được ghi log (lỗi, thử lại, truncate); judge được đối chiếu với human judge ở ý 2. Baseline:
closed-book (không truy xuất), RAG thuần + refuse theo ngưỡng cosine (ý 1), Vector RAG theo k (ý 3), xoá ngẫu nhiên /
cắt đầu cùng ngân sách token (ý 4).

## Mục lục
- [Chạy nhanh](#chạy-nhanh-sau-khi-clone-repo) · [Môi trường](#môi-trường) · [Cấu trúc repo](#cấu-trúc-repo) · [Dữ liệu](#dữ-liệu-chunks--chroma-có-trong-repo)
- [Ý 1](#ý-1--langchain-rag-thuần-vs-langgraph-self-rag) · [Ý 2](#ý-2--graphrag-vs-vector-rag-30-bài-chủ-đề-quantum) · [Ý 3](#ý-3--kg-từ-k--10-20-30-bài) · [Ý 4](#ý-4--context-compression-cho-graphrag)

## Cấu trúc repo
```
config.py, common.py          cấu hình chung (model, top-k, đường dẫn); LLM, embedding, vector store, prompt sinh câu trả lời
ingest.py                     PDF -> chunk -> Chroma (data/)
── Ý 1 ──
rag_langchain.py              hệ A: LangChain RAG thuần
rag_selfrag.py                hệ B: LangGraph Self-RAG
generate_testset.py           sinh câu hỏi (duyệt tay trong testset/review_*.py)
evaluate.py                   chấm 2 hệ bằng LLM-judge (faithfulness, answer relevancy, hallucination) + judge log
── Ý 2 ──
kg_build.py                   dựng Knowledge Graph từ 30 bài (data/kg/)
rag_graphrag.py               GraphRAG (local search qua KG)
evaluate_graphrag.py          GraphRAG vs Vector RAG (retrieval precision, generation quality)
human_judge.py                phiếu chấm tay (khuôn: testset/human_judge/template.html) + agreement human judge ↔ judge (Cohen's κ)
── Ý 3 ──
kg_quality.py                 chất lượng KG10/20/30: entity coverage, relation completeness (+ bản hiệu chỉnh)
evaluate_kg_scale.py          GraphRAG trên KG10/20/30 (--system vector_rag: Vector RAG chỉ truy xuất trong k bài)
── Ý 4 ──
compress.py                   nén context GraphRAG (KG + passage)
evaluate_compression.py       4 chế độ nén: token reduction, faithfulness (non-inferiority), context recall
                              (+ baseline xoá ngẫu nhiên / cắt đầu, hai mức nén thêm: --modes)
── Thí nghiệm bổ sung (chung cho ý 1–4) ──
run_experiments.sh            chạy toàn bộ thí nghiệm, mỗi câu trả lời chấm 3 lần bằng judge -> results/
judge_passes.py               gộp 3 lần chấm (đa số / trung bình) + độ nhất quán giữa các lần chấm
baseline_closed_book.py       baseline không truy xuất: LLM trả lời không có context
threshold_refusal.py          baseline RAG thuần + refuse theo ngưỡng cosine, đường risk–coverage
retrieval_metrics.py          chỉ số truy xuất không qua judge: gold-chunk hit, recall, MRR, nDCG@k

data/          chunks, Chroma (chroma.zip), KG (data/kg/k10, k20, k30)
papers/, papers_y2/           PDF bài báo (papers_y2: 7 bài bổ sung cho ý 2, nguồn trong SOURCES.md)
testset/       bộ câu hỏi đã duyệt tay (+ bản sinh tự động *_raw.json, kết quả duyệt review_*.py)
results/       kết quả: k4, k8, noise4 (ý 1) · kg30, human_judge (ý 2) · kg_scale (ý 3) · compression (ý 4) ·
               closed_book, retrieval_metrics.*, threshold_refusal.* (baseline); mỗi thư mục có summary.md + run_info.json,
               điểm từng lần chấm scores*.p0/p1/p2.jsonl và bản gộp scores*.jsonl; run_log.txt: nhật ký run_experiments.sh
report/        báo cáo Bao_cao_Y1..Y4 (.docx/.pdf), make_figures_y*.py (số liệu + hình), build_report_y*.py,
               stat_utils.py (bootstrap, hiệu chỉnh Holm, nhóm kết cục dùng chung)
```

## Chạy nhanh (sau khi clone repo)
Repo đã có sẵn dữ liệu (chunks + Chroma + Knowledge Graph), không cần build lại.
1. Cài [Ollama](https://ollama.com) 0.40.4 và tải 3 model:
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
   python compress.py "Which signature schemes did NIST select for standardization?" --mode both   # context nén (ý 4)
   python evaluate_compression.py                        # chạy lại đánh giá ý 4
   bash run_experiments.sh                               # toàn bộ thí nghiệm ý 1–4, chấm 3 lần (nhiều giờ, chạy tiếp được)
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
| Ollama server | 0.40.4 (mọi nội dung), `http://localhost:11434` |
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
#   -> DUYỆT TAY từng câu (đối chiếu gold chunk), kết quả duyệt ghi trong:
python testset/review_edits.py           #    single-hop -> testset/testset.json
python testset/review_multihop.py        #    multi-hop  -> testset/multihop.json
#   -> câu out-of-corpus viết tay: testset/unanswerable.json (có kiểm tra không xuất hiện trong chunks)
python evaluate.py --run k4                                                    # 3a. top-k = 4
python evaluate.py --run k8 --top-k 8 --testset testset/testset.json testset/multihop.json  # 3b. retrieval nhiễu
$env:NOISE_CHUNKS=4; python evaluate.py --run noise4 --testset testset/testset.json testset/multihop.json  # 3c. top-4 + 4 đoạn ngẫu nhiên
python baseline_closed_book.py; python threshold_refusal.py results   # 4. baseline closed-book, refuse theo ngưỡng cosine
```
Judge mặc định là `llama3.1:8b` (khác họ với LLM sinh câu trả lời để tránh tự chấm bài mình).
Đổi judge: `$env:JUDGE_MODEL='<model>'; python evaluate.py --run <tên mới>`.

## Hai hệ thống ý 1 (dùng CHUNG retriever, top-k, prompt sinh câu trả lời, LLM)
- **A. LangChain RAG** (`rag_langchain.py`): retrieve top-k → generate. 1 lần gọi LLM.
- **B. LangGraph Self-RAG** (`rag_selfrag.py`): retrieve → chấm độ liên quan từng tài liệu →
  generate → tự kiểm tra *grounded* (hallucination, prompt có few-shot) → tự kiểm tra *useful* →
  nếu sai thì sinh lại (prompt chặt hơn) / viết lại truy vấn / refuse trả lời.

## Bộ câu hỏi ý 1
| File | Loại | Số câu | Mục đích |
|---|---|---|---|
| `testset/testset.json` | single-hop: hỏi sự kiện trong 1 đoạn | 86 (sinh 93, duyệt tay) | trường hợp cơ bản |
| `testset/multihop.json` | multi-hop: cần tổng hợp 2 bài báo | 35 (sinh 71, duyệt tay) | câu hỏi khó, cần truy xuất nhiều nguồn |
| `testset/unanswerable.json` | out-of-corpus | 23 (viết tay, đã kiểm tra không có trong chunks) | hệ thống phải biết refuse (abstain) |
| (`--top-k 8`) | retrieval nhiễu | single + multi-hop | context chứa nhiều đoạn không liên quan |

## Ba chỉ số so sánh (`evaluate.py`, định nghĩa theo RAGAS)
| Chỉ số | Định nghĩa |
|---|---|
| faithfulness ↑ | #claim được context hỗ trợ / #claim |
| answer_relevancy ↑ | cosine(câu hỏi, 3 reverse-generated question từ câu trả lời); 0 nếu né tránh / refuse |
| hallucination_rate ↓ | % câu hỏi có câu trả lời chứa ≥ 1 claim không được context hỗ trợ |

Faithfulness / hallucination của **cả 2 hệ** được chấm trên cùng một context: full top-k ban đầu của câu hỏi
gốc (Self-RAG lọc bớt tài liệu trước khi sinh, nếu chấm trên tài liệu đã lọc thì không công bằng).
Secondary metric chỉ để giải thích: `*_used` (chấm trên tài liệu đưa vào generate), refusal_rate, answer_correctness,
retrieval_hit, claim_halluc_rate, latency_s, llm_calls.

Kết quả: `results/<run>/summary.md`, `summary.csv` (kèm CI 95% bootstrap), `per_question.csv`, `run_info.json`.

# Ý 2 — GraphRAG vs Vector RAG (30 bài chủ đề quantum)

Dựng Knowledge Graph (KG) từ 30 bài báo bằng LLM local, so sánh **GraphRAG** (truy xuất qua KG) với **Vector RAG**
(LangChain RAG của ý 1, embedding chunk) trên cùng 30 bài. Hai hệ dùng chung chunk, chung 4 passage mỗi câu, chung
prompt + LLM: khác biệt chỉ ở cách truy xuất. Báo cáo: `report/Bao_cao_Y2.pdf`.

**30 bài = một chủ đề chính (quantum)**, gồm 2 nhánh gần nhau để các bài có entity / khái niệm chung tạo relation:
23 bài Quantum Security + Quantum Machine Learning có trong `papers/` và 7 bài bổ sung (truy cập mở trên arXiv, bài gốc
của các khái niệm mà 23 bài kia nhắc nhiều nhất: Shor, BB84, decoy-state QKD, RSA-2048, barren plateau, VQA, NISQ),
xem `papers_y2/SOURCES.md`. **Dữ liệu ý 1 không đổi**: 7 bài nằm trong `papers_y2/` (ngoài `papers/`), chunk của chúng
trong `data/kg/chunks_y2.jsonl`, và Vector RAG của ý 2 dùng Chroma riêng `data/kg/chroma.zip` (collection `kg30`).

## Quy trình
```powershell
python kg_build.py --ingest              # 1. chia chunk 7 bài trong papers_y2/ -> data/kg/chunks_y2.jsonl
python kg_build.py --select              # 2. chọn 30 bài -> data/kg/papers.json
python kg_build.py --store               # 3. Chroma riêng của ý 2 -> data/kg/chroma.zip + MANIFEST.json
python kg_build.py --extract --k 30      # 4. LLM trích entity/relation từng chunk -> data/kg/extractions.jsonl
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
| `data/kg/papers.json` | 30 bài (Quantum Security 21, Quantum ML 9), thứ tự stratified theo chủ đề để k bài đầu giữ tỉ lệ chủ đề |
| `data/kg/extractions.jsonl` | entity + relation do qwen2.5:7b trích từ từng chunk |
| `data/kg/k30/graph.json`, `vectors.npz` | KG đã gộp (mỗi entity/relation giữ chunk nguồn) + vector entity/relation |

Bước trích xuất đặt toàn bộ LLM lên GPU (`KG_NUM_GPU = 99` trong `config.py`): mặc định Ollama chỉ đưa ~82% model
lên GPU 6 GB và chạy phần còn lại trên CPU (chậm hơn 1,5 lần). Máy ít VRAM hơn: `$env:KG_NUM_GPU='-1'`.

## Hai hệ thống ý 2
- **Vector RAG** (`rag_langchain.py`, Chroma của ý 2 qua `kg_build.use_kg_store()` trong `evaluate_graphrag.py`):
  lấy 4 chunk gần câu hỏi nhất theo cosine.
- **GraphRAG** (`rag_graphrag.py`, local search, không gọi thêm LLM): khớp câu hỏi với entity (cosine + tên xuất hiện
  nguyên văn) → 8 seed entity → mở rộng 1 bước, lấy 12 relation → chọn 4 chunk có nhiều entity/relation đã chọn.
  Context = mô tả entity + relation + 4 passage.

## Bộ câu hỏi ý 2 (85 câu, mọi câu có đáp án đều thuộc 30 bài)
| Loại | Số câu | Nguồn |
|---|---|---|
| single-hop | 27 | 21 câu của ý 1 + 6 câu mới cho bài chưa có câu (`testset/singlehop_kg.json`: sinh 7, duyệt tay) |
| multi-hop | 35 | 9 câu của ý 1 + 26 câu mới trong 30 bài (`testset/multihop_kg.json`, duyệt tay) |
| out-of-corpus | 23 | câu của ý 1 (đã kiểm tra lại từ khoá đáp án không có trong 7 bài bổ sung) |

## Chỉ số và kết quả (`results/kg30/summary.md`)
Retrieval precision = context precision (RAGAS): tỉ lệ đoạn (trong 4 đoạn) giúp suy ra ground truth. Generation
quality: faithfulness, answer relevancy, hallucination rate (như ý 1, chấm trên đúng context đưa vào LLM) + answer
correctness.

| 62 câu có đáp án | Vector RAG | GraphRAG | GraphRAG − Vector (CI 95%) |
|---|---|---|---|
| Retrieval precision ↑ | 0,63 | 0,50 | −0,129 [−0,205; −0,051] (statistically significant) |
| Faithfulness ↑ (n = 56 / 56) | 0,89 | 0,85 | −0,030 [−0,079; +0,021] (không đủ bằng chứng để phân biệt) |
| Answer relevancy ↑ | 0,74 | 0,74 | −0,003 [−0,089; +0,082] (không đủ bằng chứng để phân biệt) |
| Hallucination rate ↓ | 44% | 53% | +9,7 điểm % [−4,8; +22,6] (không đủ bằng chứng để phân biệt) |
| Answer correctness ↑ | 85% | 82% | −3,2 điểm % [−14,5; +8,1] (không đủ bằng chứng để phân biệt) |

Faithfulness chỉ tính trên câu không refuse (n = 56 mỗi hệ); các chỉ số khác tính trên đủ 62 câu.

Với cài đặt local search ở đây và KG dựng bằng mô hình 7B: GraphRAG truy xuất kém hơn chủ yếu ở câu multi-hop
(retrieval precision 0,54 so với 0,71); về generation quality không đủ bằng chứng để phân biệt hai hệ (không paired
difference nào có statistical significance, CI còn rộng); câu out-of-corpus cả hai hệ refuse đúng 23/23.
Phân tích nguyên nhân và đối chiếu tài liệu (Han et al., arXiv v3 2026; GraphRAG-Bench; HybridRAG) trong báo cáo.

## Độ tin cậy của judge (ý 2)
```powershell
python human_judge.py --make       # phiếu chấm tay testset/human_judge/phieu_cham.html (36 câu trả lời, stratified)
python human_judge.py --score      # sau khi chấm: human_labels.json -> results/human_judge/ (Cohen's κ, % agreement)
```
Mọi judge call được ghi vào `results/<run>/judge_log_*.jsonl`: status, số verdict thiếu, `prompt_eval_count`.
**Judge failure** = judge call không trả đủ verdict hợp lệ: `json_error` (JSON hỏng), `ollama_error` (Ollama huỷ lần
sinh) hoặc thiếu verdict; lần gọi lỗi được thử lại tối đa 3 lần (seed khác), verdict vẫn thiếu thì bị loại khỏi mẫu
số. Ollama **truncate** prompt dài hơn `num_ctx` mà không báo lỗi, nên prompt có độ dài đó được đánh dấu truncated.
Ý 2 (3 lần chấm, 2.871 judge call): 75 judge_failures, sau khi thử lại còn 18 lần thiếu verdict; không prompt nào bị
truncate (dài nhất 2.163 / 4.096 token). Điểm của 3 lần chấm gần như trùng hệt (temperature 0).

**Human judge** (18 câu hỏi stratified × 2 hệ = 36 câu trả lời, phiếu chấm mù): agreement với judge thấp — Cohen's κ
0,13 (passage hữu ích), 0,15 (answer correctness), 0,28 (hallucination). Judge dễ dãi hơn ở bước chấm passage và
answer correctness (83% câu đúng so với 42% của human judge) nhưng khắt khe hơn ở bước verify claim (faithfulness 0,869
so với 0,962). Chi tiết: `results/human_judge/human_judge.md`, báo cáo Y2 mục 8.5.

# Ý 3 — KG từ k = 10, 20, 30 bài

Dựng KG từ **10, 20 và 30 bài đầu** của `data/kg/papers.json` (30 bài chủ đề quantum của ý 2, thứ tự stratified theo
chủ đề cố định từ ý 2), đo **chất lượng KG** (entity coverage, relation completeness) và **chất lượng câu trả lời** của
GraphRAG (ý 2) trên từng KG. Ba KG dùng chung cache trích xuất của ý 2 nên **lồng nhau** (KG10 ⊂ KG20 ⊂ KG30): khác
biệt chỉ đến từ số bài. Báo cáo: `report/Bao_cao_Y3.pdf`. Ý 3 chỉ thêm file mới, không sửa mã / dữ liệu của ý 1, ý 2.

## Quy trình
```powershell
python kg_build.py --build --k 10        # 1. KG10 -> data/kg/k10/ (gộp lại từ data/kg/extractions.jsonl, không trích lại)
python kg_build.py --build --k 20        #    KG20 -> data/kg/k20/   (KG30 = data/kg/k30/ của ý 2)
python kg_quality.py --gold              # 2. judge trích entity / gold relation của 62 câu -> testset/kg_gold_raw.json
python testset/review_kg_gold.py         #    kết quả duyệt tay -> testset/kg_gold.json
python kg_quality.py --all               # 3. chất lượng KG -> results/kg_scale/kg_quality.md (tất định, không gọi LLM)
python evaluate_kg_scale.py              # 4. GraphRAG với KG10/20/30 trên 85 câu của ý 2 -> results/kg_scale/summary.md
python evaluate_kg_scale.py --system vector_rag   #    Vector RAG chỉ truy xuất trong k bài (so sánh theo k)
python report/make_figures_y3.py; python report/build_report_y3.py   # 5. báo cáo -> report/Bao_cao_Y3.docx/.pdf
```
Các bước 1–4 đã chạy sẵn, kết quả có trong repo. Kết quả GraphRAG với KG30 lấy lại từ `results/kg30/` của ý 2 (cùng mã,
KG, câu hỏi, môi trường).

## Dữ liệu ý 3 (có trong repo)
| File | Nội dung |
|---|---|
| `data/kg/k10/`, `data/kg/k20/` | KG10 (3.274 entity, 3.252 relation), KG20 (5.702 / 6.160): `graph.json` + `vectors.npz` |
| `testset/kg_gold.json` | gold set: 62 câu, 353 entity, 263 relation (bản tự động `kg_gold_raw.json`: 503 / 415; duyệt tay trong `review_kg_gold.py`) |
| `results/kg_scale/` | chất lượng KG (`kg_quality.md`, `kg_coverage*.csv`, `kg_growth.csv`, `kg_relation_audit.csv`) + câu trả lời theo k (`k10/`, `k20/`, `k30/`, `summary.md`) |

## Cách đo
- **Entity coverage**: tỉ lệ gold entity có trong KG, khớp theo tên sau chuẩn hoá (cùng quy tắc gộp entity của
  `kg_build.py`); bản `_lenient` (upper bound) tính thêm entity có tên chứa trọn cụm từ chuẩn. Không dùng LLM để khớp
  (đã thử llama3.1:8b, qwen2.5:7b: nhận nhầm khái niệm rộng / hẹp hơn).
- **Relation completeness**: tỉ lệ gold relation có edge trực tiếp trong KG nối 2 entity đã khớp; audit 40
  edge KG30: 36 edge nói đúng gold relation. Bản **hiệu chỉnh** = relation completeness × 36/40, CI 95% bootstrap gộp
  sai số lấy mẫu gold relation và sai số của tỉ lệ audit (`relation_completeness_adj` trong `kg_coverage.csv`).
- **Câu trả lời**: 5 chỉ số của ý 2, xem theo 3 tập: *in-scope* (bài nguồn thuộc k bài: 15 / 37 / 62 câu),
  *fixed set* S10 (15 câu về 10 bài đầu) và S20 (37 câu về 20 bài đầu) paired comparison, *toàn bộ 62 câu*.

## Kết quả
| | KG10 | KG20 | KG30 |
|---|---|---|---|
| Entity coverage (câu in-scope) | 91% | 90% | 89% |
| Relation completeness (câu in-scope) | 39% | 37% | 42% |
| Relation completeness hiệu chỉnh × 0,9 theo audit [CI 95%] | 35% [24; 47] | 33% [26; 42] | 37% [31; 44] |
| Entity coverage / relation completeness (toàn bộ 62 câu) | 64% / 20% | 77% / 31% | 89% / 42% |
| Answer correctness (câu in-scope) | 80% | 86% | 82% |
| Answer correctness (toàn bộ 62 câu) | 53% | 76% | 82% |
| Vector RAG chỉ truy xuất trong k bài: answer correctness (toàn bộ 62 câu) | 52% | 79% | 85% |
| Câu out-of-corpus refuse đúng | 23/23 | 23/23 | 23/23 |

- Với câu hỏi về chính các bài trong KG: **không phát hiện khác biệt trên S10 / S20; năng lực kiểm định thấp** (S10
  chỉ 15 câu: chênh lệch answer correctness KG10 − KG30 là +0,0 điểm % [−20,0; +20,0]), nên chỉ loại trừ được chênh
  lệch lớn.
- Điểm yếu chung là **relation** (~40% relation cần thiết có trong KG), do mô hình trích xuất 7B chứ không do số bài.
- KG **chưa đạt saturation**: mỗi bài thêm vào vẫn mang ~250–300 entity mới; trên toàn bộ 62 câu KG10 kém KG30 có statistical
  significance (answer correctness −29,0 điểm % [−41,9; −16,1]), còn KG20 thì không đủ bằng chứng để phân biệt với KG30 (−6,5 điểm % [−16,1; +1,6]).
- **"Đủ" theo tiêu chí non-inferiority** (answer correctness so với KG30, δ = 10 điểm %, đặt khi rà soát lại báo cáo):
  KG20 đủ cho câu về 20 bài đầu; KG10 chưa kết luận được cho câu về 10 bài đầu (S10 chỉ 15 câu); với câu về cả 30 bài,
  KG10 không đủ, KG20 chưa kết luận được. Câu out-of-scope vẫn được trả lời đúng 45% (KG10) / 60% (KG20), chủ
  yếu nhờ kiến thức sẵn có của mô hình (claim không có trong context).

- **So với Vector RAG chỉ truy xuất trong cùng k bài**: ở mọi k không đủ bằng chứng để phân biệt answer correctness;
  retrieval precision của GraphRAG thấp hơn (statistically significant ở k = 20, 30). Vector RAG cũng giảm mạnh khi
  chỉ có 10 bài, nên sự phụ thuộc vào phạm vi tài liệu không riêng của KG.

**Judge ý 3** (3 lần chấm, 5.068 judge call): 85 judge_failures, sau khi thử lại còn 18 lần thiếu verdict, không prompt
nào bị truncate; điểm GraphRAG của 3 lần chấm gần như trùng hệt. "Faithfulness KG10 < KG30" có CI không chứa 0 nhưng
không qua hiệu chỉnh Holm, nên không dùng làm kết luận.

Đối chiếu tài liệu trong báo cáo: Han et al. (arXiv v3, 2026), GraphRAG-Bench (thử nghiệm theo cỡ corpus), KGGen / MINE,
Zhu et al., Joren et al. (sufficient context, ICLR 2025), định luật Heaps.

# Ý 4 — Context compression cho GraphRAG

Nén context mà GraphRAG (ý 2, KG30) đưa vào LLM — mô tả KG + 4 passage — để giảm token, đo **token reduction** (mục
tiêu 19–53%) và kiểm tra **faithfulness có giữ nguyên không**. Báo cáo: `report/Bao_cao_Y4.pdf`. Ý 4 chỉ thêm file mới.

## Quy trình
```powershell
python compress.py --calibrate          # 1. chọn mức nén CHỈ theo độ dài context (không gọi LLM)
python compress.py "câu hỏi" --mode both  #    xem context nén (base / kg / text / both)
python evaluate_compression.py          # 2. 4 chế độ x 85 câu: sinh câu trả lời, chấm -> results/compression/
python evaluate_compression.py --modes rand_both rand_both_s1 rand_both_s2 head_both rand_kg head_kg rand_text head_text both_l25 both_l40
                                        #    baseline cùng ngân sách (xoá ngẫu nhiên / cắt đầu) + hai mức nén thêm
python report/make_figures_y4.py; python report/build_report_y4.py   # 3. báo cáo -> report/Bao_cao_Y4.docx/.pdf
```

## Cách nén (`compress.py`) — chỉ xoá bớt (extractive), context nén luôn là tập con của context gốc
- **Nén KG**: giữ 50% entity / 50% relation gần câu hỏi nhất (cosine embedding); mỗi dòng chỉ giữ mảnh mô tả gần câu
  hỏi nhất (≤ 110 ký tự, lấy trong phần mô tả context gốc hiển thị).
- **Nén passage**: bỏ nhiễu (số trích dẫn, URL, email, dòng arXiv), giữ 50% số câu gần câu hỏi nhất mỗi passage.
- Bốn chế độ: base (không nén), kg, text, both. Cả bốn sinh câu trả lời lại trong cùng môi trường (cả base).

## Cách đo (`evaluate_compression.py`)
- **Token reduction**: 1 − token context nén / token context gốc, token đếm bằng `prompt_eval_count` (tokenizer qwen2.5:7b).
- **Faithfulness** trên context đưa vào LLM; "giữ nguyên" kiểm định **non-inferiority**, ngưỡng đặt trước 0,05.
- **Context recall** (claim của ground truth còn trong context: "thông tin quan trọng"), answer correctness, refusal
  ở câu có đáp án, answer relevancy; judge log (judge_failures, truncate) như ý 2.

## Kết quả (62 câu có đáp án; nén − gốc, CI 95%)
| | Nén KG | Nén passage | Nén KG + passage |
|---|---|---|---|
| Token reduction | 22,5% | 25,3% | 47,9% |
| Faithfulness | −0,026 [−0,070; +0,018] | −0,025 [−0,074; +0,020] | −0,066 [−0,120; −0,016] |
| Context recall | −0,7 điểm % [−3,7; +2,3] | −2,2 điểm % [−5,7; +1,2] | −3,2 điểm % [−6,6; +0,3] |
| Answer correctness | −3,2 điểm % [−11,3; +4,8] | −1,6 điểm % [−8,1; +3,2] | −1,6 điểm % [−9,7; +6,5] |
| Refusal (câu có đáp án) | +8,1 điểm % [+1,6; +16,1] | +3,2 điểm % [0,0; +8,1] | +8,1 điểm % [+1,6; +16,1] |

- Token reduction của cả ba mức nằm trong mục tiêu 19–53%.
- **Không chế độ nào chứng minh được faithfulness giữ nguyên** (non-inferiority, ngưỡng 0,05): nén một phần giảm ~0,03,
  không có statistical significance nhưng không đủ bằng chứng để nói giữ nguyên (năng lực kiểm định thấp; chưa loại
  trừ được mức giảm tới 0,070–0,074); nén cả hai giảm 0,066 [−0,120; −0,016], CI không chứa 0 nhưng không qua hiệu
  chỉnh Holm cho 21 so sánh. Hai mức nén thêm cho chế độ nén cả hai (token reduction 27%, 39%): faithfulness giảm dần
  khi nén mạnh hơn, ngay ở mức nhẹ nhất vẫn chưa chứng minh được giữ nguyên.
- Context recall và answer correctness: không chế độ nào có dấu hiệu giảm rõ. Nén KG làm **refusal ở câu có đáp án
  tăng gấp đôi** (8% → 16%, chưa qua hiệu chỉnh Holm), ở cả ba cách chọn dòng KG, chủ yếu câu multi-hop. Không paired
  difference nào còn statistical significance sau hiệu chỉnh Holm.
- So với xoá ngẫu nhiên / cắt đầu ở cùng ngân sách token: chọn theo độ liên quan giữ context recall tốt hơn ở phần
  passage (khoảng +5 điểm %, CI không chứa 0, chưa qua hiệu chỉnh Holm); ở phần mô tả KG cách chọn gần như không quan trọng.
- Judge (3 lần chấm, 13.455 judge call): 88 judge_failures, sau khi thử lại còn 18 lần thiếu verdict, không prompt nào
  bị truncate; độ lệch của chênh lệch faithfulness giữa các lần chấm (SD ≈ 0,004) nhỏ hơn nhiều so với biên 0,05.

Đối chiếu tài liệu trong báo cáo: KG-RAG (Soman et al., Bioinformatics 2024), RECOMP (ICLR 2024), LongLLMLingua (ACL 2024).
