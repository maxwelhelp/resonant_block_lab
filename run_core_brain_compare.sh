#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_BASE="${OUT_BASE:-runs/core_brain_compare_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-10}"
TRAIN_N="${TRAIN_N:-12000}"
VAL_N="${VAL_N:-2000}"
LOG_EVERY="${LOG_EVERY:-50}"
mkdir -p "$OUT_BASE"
REPORT="$OUT_BASE/FINAL_CORE_REPORT.md"

echo "# Core Brain Compare" > "$REPORT"
echo "out: $OUT_BASE" >> "$REPORT"
echo "date: $(date -Is)" >> "$REPORT"

run() {
  local name="$1"; shift
  local log="$OUT_BASE/${name}.log"
  echo "[RUN] $name"
  echo "\n## $name" >> "$REPORT"
  "$@" >"$log" 2>&1
  rc=$?
  echo "returncode: $rc" >> "$REPORT"
  if [ $rc -ne 0 ]; then
    echo '```text' >> "$REPORT"
    tail -n 100 "$log" >> "$REPORT"
    echo '```' >> "$REPORT"
    echo "[FAIL] $name"
    exit $rc
  fi
  echo "[OK] $name"
}

summary() {
  local label="$1"; local dir="$2"
  echo "\n## Summary $label" >> "$REPORT"
  "$PYTHON_BIN" - <<PY >> "$REPORT" 2>&1
import csv, json, pathlib
p=pathlib.Path('$dir')
print('dir:', p)
h=p/'history.csv'
if h.exists():
    rows=list(csv.DictReader(open(h)))
    print('epochs:', len(rows))
    if rows:
        best=max(float(r.get('val_acc','nan')) for r in rows)
        last=rows[-1]
        print('best_val_acc:', round(best,6))
        for k in ['epoch','train_acc','val_acc','val_macc','val_uacc','val_alpha','val_ce','val_macro_ce','val_micro_ce']:
            if k in last: print(k, last[k])
else:
    print('history.csv missing')
for fn in ['program_summary.json','best_program.json']:
    f=p/fn
    print(fn, f.exists())
    if f.exists():
        s=json.load(open(f))
        tc=s.get('true_counterfactual_credit',{})
        print('final_acc:', s.get('final_acc'))
        print('policy:', s.get('policy_action',{}).get('action'))
        print('true_credit:', tc.get('enabled'), 'rows:', len(tc.get('rows',[])))
        print('helpful_true:', tc.get('helpful_modes_true',[])[:3])
        print('pair_top:', s.get('pair_compatibility_proxy_top',[])[:3])
PY
}

run compile "$PYTHON_BIN" -m py_compile \
  resonant_block_lab/resonant_block.py \
  resonant_block_lab/fractal_sequence.py \
  resonant_block_lab/program_brain.py \
  train_fractal_resonant.py \
  train_attention_swap.py

run probe_topk_act "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs 2 --batch 64 --dim 32 --n_modes 8 --macro_steps 3 --micro_steps 2 \
  --train_n 512 --val_n 256 --device cuda --program_action_mode act --mix_topk 3 \
  --out_dir "$OUT_BASE/probe_topk_act"
summary probe_topk_act "$OUT_BASE/probe_topk_act"

run dense_diagnostic "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs "$EPOCHS" --batch 128 --length 128 --dim 64 --classes 8 --n_modes 8 \
  --macro_steps 8 --micro_steps 3 --train_n "$TRAIN_N" --val_n "$VAL_N" --noise 0.24 \
  --device cuda --program_action_mode diagnostic --mix_topk 0 --log_every "$LOG_EVERY" \
  --out_dir "$OUT_BASE/dense_diagnostic"
summary dense_diagnostic "$OUT_BASE/dense_diagnostic"

run dense_active "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs "$EPOCHS" --batch 128 --length 128 --dim 64 --classes 8 --n_modes 8 \
  --macro_steps 8 --micro_steps 3 --train_n "$TRAIN_N" --val_n "$VAL_N" --noise 0.24 \
  --device cuda --program_action_mode act --mix_topk 0 --log_every "$LOG_EVERY" \
  --out_dir "$OUT_BASE/dense_active"
summary dense_active "$OUT_BASE/dense_active"

run topk3_active "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs "$EPOCHS" --batch 128 --length 128 --dim 64 --classes 8 --n_modes 8 \
  --macro_steps 8 --micro_steps 3 --train_n "$TRAIN_N" --val_n "$VAL_N" --noise 0.24 \
  --device cuda --program_action_mode act --mix_topk 3 --log_every "$LOG_EVERY" \
  --out_dir "$OUT_BASE/topk3_active"
summary topk3_active "$OUT_BASE/topk3_active"

echo "\n# Done" >> "$REPORT"
echo "REPORT=$REPORT"
