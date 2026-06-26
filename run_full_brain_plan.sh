#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_BASE="${OUT_BASE:-runs/full_brain_plan_$(date +%Y%m%d_%H%M%S)}"
export OUT_BASE
mkdir -p "$OUT_BASE"
REPORT="$OUT_BASE/FINAL_PLAN_REPORT.md"

run_cmd() {
  local name="$1"; shift
  local log="$OUT_BASE/${name}.log"
  echo "\n## $name" | tee -a "$REPORT"
  echo '```bash' >> "$REPORT"
  printf '%q ' "$@" >> "$REPORT"
  echo -e '\n```' >> "$REPORT"
  echo "[RUN] $name"
  "$@" >"$log" 2>&1
  local rc=$?
  echo "returncode: $rc" | tee -a "$REPORT"
  if [ $rc -ne 0 ]; then
    echo '```text' >> "$REPORT"
    tail -n 80 "$log" >> "$REPORT"
    echo '```' >> "$REPORT"
    echo "[FAIL] $name, see $log"
    return $rc
  fi
  echo "[OK] $name"
  return 0
}

summarize_run() {
  local label="$1"; local run_dir="$2"
  echo "\n### Summary: $label" >> "$REPORT"
  "$PYTHON_BIN" - <<PY >> "$REPORT" 2>&1
import csv, json, pathlib
p=pathlib.Path('$run_dir')
h=p/'history.csv'
print('run_dir:', p)
if h.exists():
    rows=list(csv.DictReader(open(h)))
    print('epochs:', len(rows))
    if rows:
        best=max(float(r.get('val_acc', 'nan')) for r in rows)
        r=rows[-1]
        keys=['epoch','train_acc','val_acc','val_macc','val_uacc','val_alpha','val_ce','val_macro_ce','val_micro_ce']
        print('best_val_acc:', round(best, 6))
        print('last:', {k:r.get(k) for k in keys if k in r})
else:
    print('history.csv missing')
for name in ['program_summary.json','best_program.json']:
    f=p/name
    print(name, 'exists=', f.exists())
    if f.exists():
        s=json.load(open(f))
        tc=s.get('true_counterfactual_credit', {})
        print(' final_acc=', s.get('final_acc'), 'policy=', s.get('policy_action',{}).get('action'))
        print(' true_credit=', tc.get('enabled'), 'rows=', len(tc.get('rows',[])))
        print(' helpful_true=', tc.get('helpful_modes_true', [])[:3])
        print(' helpful_proxy=', s.get('helpful_modes_proxy', [])[:3])
        print(' pair_top=', s.get('pair_compatibility_proxy_top', [])[:3])
PY
}

echo "# Full Brain Plan Report" > "$REPORT"
echo "root: $ROOT" >> "$REPORT"
echo "out: $OUT_BASE" >> "$REPORT"
echo "date: $(date -Is)" >> "$REPORT"

run_cmd compile "$PYTHON_BIN" -m py_compile \
  resonant_block_lab/resonant_block.py \
  resonant_block_lab/fractal_sequence.py \
  resonant_block_lab/program_brain.py \
  train_fractal_resonant.py \
  train_attention_swap.py || exit 1

# Fast probes first.
run_cmd probe_fractal_topk_active "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs 2 --batch 64 --dim 32 --n_modes 8 --macro_steps 3 --micro_steps 2 \
  --train_n 512 --val_n 256 --device cuda --program_action_mode act --mix_topk 3 \
  --out_dir "$OUT_BASE/probe_fractal_topk_active" || exit 1
summarize_run probe_fractal_topk_active "$OUT_BASE/probe_fractal_topk_active"

run_cmd probe_swap_topk "$PYTHON_BIN" -u - <<'PYRUN'
import os, runpy, sys
sys.argv=['train_attention_swap.py','--variant','res_instead_attn','--epochs','1','--batch','64','--dim','32','--n_modes','8','--macro_steps','3','--micro_steps','2','--train_n','256','--val_n','128','--device','cuda','--mix_topk','3','--out_dir','__OUT_BASE__/probe_swap_topk']
runpy.run_path('train_attention_swap.py', run_name='__main__')
PYRUN
# Fix placeholder in generated log command not needed for execution; direct attention command below is safer for normal runs.

# Full comparison. Set QUICK=1 for 3 epoch smoke.
if [ "${QUICK:-0}" = "1" ]; then
  EPOCHS=3; TRAIN_N=4000; VAL_N=1000; LOG_EVERY=25
else
  EPOCHS=${EPOCHS:-10}; TRAIN_N=${TRAIN_N:-12000}; VAL_N=${VAL_N:-2000}; LOG_EVERY=${LOG_EVERY:-50}
fi

run_cmd dense_diagnostic "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs "$EPOCHS" --batch 128 --length 128 --dim 64 --classes 8 --n_modes 8 \
  --macro_steps 8 --micro_steps 3 --train_n "$TRAIN_N" --val_n "$VAL_N" --noise 0.24 \
  --device cuda --program_action_mode diagnostic --mix_topk 0 --log_every "$LOG_EVERY" \
  --out_dir "$OUT_BASE/dense_diagnostic" || exit 1
summarize_run dense_diagnostic "$OUT_BASE/dense_diagnostic"

run_cmd dense_active "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs "$EPOCHS" --batch 128 --length 128 --dim 64 --classes 8 --n_modes 8 \
  --macro_steps 8 --micro_steps 3 --train_n "$TRAIN_N" --val_n "$VAL_N" --noise 0.24 \
  --device cuda --program_action_mode act --mix_topk 0 --log_every "$LOG_EVERY" \
  --out_dir "$OUT_BASE/dense_active" || exit 1
summarize_run dense_active "$OUT_BASE/dense_active"

run_cmd topk3_active "$PYTHON_BIN" -u train_fractal_resonant.py \
  --epochs "$EPOCHS" --batch 128 --length 128 --dim 64 --classes 8 --n_modes 8 \
  --macro_steps 8 --micro_steps 3 --train_n "$TRAIN_N" --val_n "$VAL_N" --noise 0.24 \
  --device cuda --program_action_mode act --mix_topk 3 --log_every "$LOG_EVERY" \
  --out_dir "$OUT_BASE/topk3_active" || exit 1
summarize_run topk3_active "$OUT_BASE/topk3_active"

echo "\n# Done" >> "$REPORT"
echo "report: $REPORT"
