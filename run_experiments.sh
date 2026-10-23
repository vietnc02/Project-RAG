#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm của nội dung 1–4 (kết quả trong results/r2/), mỗi câu trả lời chấm 3 lượt judge.
#
#   bash run_experiments.sh       # chạy tiếp từ chỗ đã dừng (mọi bước đều bỏ qua câu đã có kết quả)
#
# Môi trường: Ollama 0.40.4 (`ollama serve`), qwen2.5:7b, llama3.1:8b, nomic-embed-text (xem README).
# Thứ tự: lượt chấm 0 cho mọi thí nghiệm trước, sau đó lượt 1–2 (seed judge khác) theo thứ tự ưu tiên. Sau mỗi bước,
# judge_passes.py gộp các lượt đã có thành scores_*.jsonl, nên có thể dừng bất cứ lúc nào và dùng kết quả đã có.
set -u
cd "$(dirname "$0")"
PY=${PY:-D:/RAG_env/venv/Scripts/python.exe}
export RESULTS_DIR=results/r2 PYTHONIOENCODING=utf-8
R=$RESULTS_DIR
mkdir -p "$R"
LOG="$R/run_log.txt"

want=$(curl -s localhost:11434/api/version)
case "$want" in *0.40.4*) ;; *) echo "Ollama không phải 0.40.4: $want" | tee -a "$LOG"; exit 1 ;; esac

step() {  # step <tên> <lượt> <lệnh...>
  local name=$1 pass=$2; shift 2
  echo "[$(date '+%F %T')] BẮT ĐẦU $name (lượt $pass)" >> "$LOG"
  JUDGE_PASS=$pass "$@" > "$R/last_step.out" 2>&1
  local rc=$?
  tail -n 3 "$R/last_step.out" >> "$LOG"
  echo "[$(date '+%F %T')] XONG $name (lượt $pass, mã $rc)" >> "$LOG"
  "$PY" judge_passes.py "$R" >> "$LOG" 2>&1
}

Y4_BASELINES="rand_both rand_both_s1 rand_both_s2 head_both rand_kg head_kg rand_text head_text both_l25 both_l40"
SINGLE_MH="testset/testset.json testset/multihop.json"

# ---------- lượt chấm 0: mọi thí nghiệm ----------
step "Y2 kg30"            0 "$PY" evaluate_graphrag.py --run kg30
step "Y3 GraphRAG k"      0 "$PY" evaluate_kg_scale.py
step "Y4 nén (4 chế độ)"  0 "$PY" evaluate_compression.py
step "Y1 k4"              0 "$PY" evaluate.py --run k4
step "X2 closed-book"     0 "$PY" baseline_closed_book.py
step "Y3 Vector RAG k"    0 "$PY" evaluate_kg_scale.py --system vector_rag
step "Y1 k8"              0 "$PY" evaluate.py --run k8 --top-k 8 --testset $SINGLE_MH
step "Y1 nhiễu thuần"     0 env NOISE_CHUNKS=4 "$PY" evaluate.py --run noise4 --testset $SINGLE_MH
step "Y4 baseline nén"    0 "$PY" evaluate_compression.py --modes $Y4_BASELINES

# ---------- lượt chấm 1, 2 (đo nhiễu của judge, lấy đa số) theo thứ tự ưu tiên ----------
for pass in 1 2; do
  step "Y4 nén (4 chế độ)" $pass "$PY" evaluate_compression.py
  step "Y2 kg30"           $pass "$PY" evaluate_graphrag.py --run kg30
  step "Y3 GraphRAG k"     $pass "$PY" evaluate_kg_scale.py
  step "Y1 k4"             $pass "$PY" evaluate.py --run k4
done
for pass in 1 2; do
  step "Y3 Vector RAG k"   $pass "$PY" evaluate_kg_scale.py --system vector_rag
  step "Y1 k8"             $pass "$PY" evaluate.py --run k8 --top-k 8 --testset $SINGLE_MH
  step "Y1 nhiễu thuần"    $pass env NOISE_CHUNKS=4 "$PY" evaluate.py --run noise4 --testset $SINGLE_MH
  step "X2 closed-book"    $pass "$PY" baseline_closed_book.py
  step "Y4 baseline nén"   $pass "$PY" evaluate_compression.py --modes $Y4_BASELINES
done

# ---------- tổng hợp (không gọi LLM sinh câu trả lời) ----------
"$PY" judge_passes.py "$R" >> "$LOG" 2>&1
"$PY" evaluate_graphrag.py --run kg30 --report-only >> "$LOG" 2>&1
"$PY" evaluate_kg_scale.py --report-only >> "$LOG" 2>&1
"$PY" evaluate_compression.py --report-only >> "$LOG" 2>&1
for run in k4 k8 noise4; do
  args=""; [ $run != k4 ] && args="--testset $SINGLE_MH"; [ $run = k8 ] && args="$args --top-k 8"
  "$PY" evaluate.py --run $run $args --report-only >> "$LOG" 2>&1
done
"$PY" retrieval_metrics.py "$R" >> "$LOG" 2>&1
"$PY" threshold_refusal.py "$R" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] HOÀN TẤT" >> "$LOG"
