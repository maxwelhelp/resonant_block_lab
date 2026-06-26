#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=.
RUN_ROOT="${RUN_ROOT:-runs/token_lm_sweep_$(date +%Y%m%d_%H%M%S)}"
CORPUS_DIR="${CORPUS_DIR:-/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST}"
mkdir -p "$RUN_ROOT"
for name in res_discovery res_memory_v2; do
  echo "================ $name ================"
  if [ "$name" = "res_discovery" ]; then
    FLAGS="--variant res_lm --n_modes 10 --enable_token_primitives --enable_matrix_program --enable_lowrank --enable_input_primitive --program_steps 2 --program_rank 8"
  else
    FLAGS="--variant res_lm --n_modes 10 --enable_token_primitives --enable_matrix_program --enable_lowrank --enable_input_primitive --enable_slot_memory --memory_slots ${MEMORY_SLOTS:-6} --w_mem_read ${W_MEM_READ:-0.005} --w_mem_write ${W_MEM_WRITE:-0.010} --w_mem_garbage ${W_MEM_GARBAGE:-0.080} --w_mem_energy ${W_MEM_ENERGY:-0.001} --program_steps 2 --program_rank 8"
  fi
  PYTHONPATH=. python -u train_token_lm_swap.py \
    --vocab "${VOCAB:-256}" \
    --task "${TASK:-byte_lm}" \
    --memory_delay "${MEMORY_DELAY:-64}" \
    --memory_pairs "${MEMORY_PAIRS:-3}" \
    --epochs "${EPOCHS:-10}" \
    --batch "${BATCH:-128}" \
    --seq_len "${SEQ_LEN:-128}" \
    --dim "${DIM:-128}" \
    --macro_steps "${MACRO:-6}" \
    --micro_steps "${MICRO:-2}" \
    --train_samples "${TRAIN_SAMPLES:-20000}" \
    --val_samples "${VAL_SAMPLES:-2000}" \
    --max_bytes "${MAX_BYTES:-20000000}" \
    --corpus_dir "$CORPUS_DIR" \
    --out_dir "$RUN_ROOT/$name" \
    --credit_modes "${CREDIT_MODES:-0}" \
    --log_every "${LOG_EVERY:-50}" \
    $FLAGS 2>&1 | tee "$RUN_ROOT/${name}.log"
done
echo "DONE $RUN_ROOT"
