#!/bin/bash
#SBATCH --job-name=merit-regression
#SBATCH --output=logs/merit_regression_%j.log
#SBATCH --partition=hpg-default
#SBATCH --mem=48gb
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# MERIT Geographic Regression Test
#
# Runs the MERIT validation pipeline (stages 2-4) with the UTM-aware
# pysheds fork on PYTHONPATH. Confirms geographic CRS code path is
# unchanged after Phase A modifications.
#
# Pass criteria: stage4 output must match the previous audit baseline
# (audit/240210-.../claude_merit_validation_audit/stage4/stage4_results.json).
# Tolerance: 0.01 on correlation values.
#
# Note: stage4 has a known pre-existing bug — it compares aspect in
# degrees (ours) vs radians (published) using Pearson correlation,
# giving 0.6487 instead of the true 0.9999 circular correlation
# (computed separately by stage5). This is NOT a regression; the
# baseline value 0.6487 is what stage4 has always produced.
#
# Expected stage4 correlations (from previous audit baseline):
#   Height (HAND):    0.9999
#   Distance (DTND):  0.9982
#   Slope:            0.9966
#   Aspect (Pearson): 0.6487  (known bug: radians vs degrees)
#   Width:            0.9597
#   Area fraction:    0.8200
#
# Usage:
#   cd $SWENSON
#   sbatch scripts/smoke_tests/run_merit_regression.sh

set -euo pipefail

SWENSON="/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson"
PYSHEDS_FORK="/blue/gerber/cdevaneprugh/pysheds_fork"

cd "$SWENSON"

# --- Environment setup ---
module load conda 2>/dev/null
conda activate ctsm

# Use the UTM-aware pysheds fork (audit/pgrid-and-tests branch)
export PYTHONPATH="${PYSHEDS_FORK}:${PYTHONPATH:-}"

echo "=== MERIT Geographic Regression Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Verify pysheds fork branch
FORK_BRANCH=$(cd "$PYSHEDS_FORK" && git branch --show-current)
echo "pysheds fork branch: $FORK_BRANCH"
if [[ "$FORK_BRANCH" != "audit/pgrid-and-tests" ]]; then
    echo "WARNING: Expected branch 'audit/pgrid-and-tests', got '$FORK_BRANCH'"
fi
echo ""

# Verify pysheds imports from fork
python -c "
import pysheds.pgrid as pg
import os
fork_path = os.path.dirname(os.path.dirname(pg.__file__))
print(f'pysheds loaded from: {fork_path}')
assert 'pysheds_fork' in fork_path, f'pysheds not loading from fork: {fork_path}'
print('OK: pysheds loading from fork')
"
echo ""

# --- Stage 2: Spatial Scale (FFT) ---
echo "=== Stage 2: Spatial Scale Analysis ==="
echo "Start: $(date)"
python scripts/merit_validation/stage2_spatial_scale.py
echo "End: $(date)"
echo ""

# Quick check: verify stage 2 output exists
if [[ ! -f output/merit_validation/stage2/stage2_results.json ]]; then
    echo "FAIL: stage2_results.json not created"
    exit 1
fi
echo "Stage 2 output:"
python -c "
import json
with open('output/merit_validation/stage2/stage2_results.json') as f:
    d = json.load(f)
be = d.get('best_estimate', d)
print(f'  Lc = {be.get(\"lc_meters\", \"N/A\")} m')
print(f'  A_thresh = {be.get(\"accum_threshold_cells\", \"N/A\")} cells')
"
echo ""

# --- Stage 3: Hillslope Parameters ---
echo "=== Stage 3: Hillslope Parameter Computation ==="
echo "Start: $(date)"
python scripts/merit_validation/stage3_hillslope_params.py
echo "End: $(date)"
echo ""

# Quick check: verify stage 3 output exists
if [[ ! -f output/merit_validation/stage3/stage3_hillslope_params.json ]]; then
    echo "FAIL: stage3_hillslope_params.json not created"
    exit 1
fi
echo "Stage 3 complete."
echo ""

# --- Stage 4: Comparison to Published Data ---
echo "=== Stage 4: Comparison to Published Data ==="
echo "Start: $(date)"
python scripts/merit_validation/stage4_comparison.py
echo "End: $(date)"
echo ""

# Quick check: verify stage 4 output exists
if [[ ! -f output/merit_validation/stage4/stage4_results.json ]]; then
    echo "FAIL: stage4_results.json not created"
    exit 1
fi

# --- Stage 5: Circular Aspect Correlation ---
echo "=== Stage 5: Circular Aspect Correlation ==="
echo "Start: $(date)"
python scripts/merit_validation/stage5_unit_fix.py
echo "End: $(date)"
echo ""

# Quick check: verify stage 5 output exists
if [[ ! -f output/merit_validation/stage5/stage5_results.json ]]; then
    echo "FAIL: stage5_results.json not created"
    exit 1
fi
echo "Stage 5 complete."
echo ""

# --- Results Summary ---
echo "========================================"
echo "=== REGRESSION TEST RESULTS ==="
echo "========================================"
python -c "
import json, sys

with open('output/merit_validation/stage4/stage4_results.json') as f:
    data = json.load(f)

if data.get('status') != 'comparison_complete':
    print(f'Stage 4 status: {data.get(\"status\", \"unknown\")}')
    print('RESULT: FAIL — comparison did not complete')
    sys.exit(1)

metrics = data['metrics']

# Expected correlations from previous audit baseline
# (audit/240210-.../claude_merit_validation_audit/stage4/stage4_results.json)
#
# Aspect uses the raw stage4 Pearson value (0.6487), NOT the
# corrected circular value (0.9999) from stage5. Stage4 has a
# known pre-existing bug comparing degrees vs radians. We test
# against what stage4 actually produces to detect regressions.
expected = {
    'height': 0.9999,
    'distance': 0.9982,
    'slope': 0.9966,
    'aspect': 0.6487,
    'width': 0.9597,
    'area': 0.8200,
}

# Display names for readability
display = {
    'height': 'Height (HAND)',
    'distance': 'Distance (DTND)',
    'slope': 'Slope',
    'aspect': 'Aspect (Pearson*)',
    'width': 'Width',
    'area': 'Area fraction',
}

TOLERANCE = 0.01
any_fail = False

print(f'{\"Parameter\":<25} {\"Expected\":>10} {\"Actual\":>10} {\"Delta\":>10} {\"Status\":>8}')
print('-' * 67)

for param, exp in expected.items():
    if param not in metrics:
        print(f'{display[param]:<25} {exp:>10.4f} {\"N/A\":>10} {\"\":>10} {\"MISSING\":>8}')
        any_fail = True
        continue

    actual = metrics[param]['correlation']
    delta = actual - exp
    status = 'PASS' if abs(delta) <= TOLERANCE else 'FAIL'
    if status == 'FAIL':
        any_fail = True
    print(f'{display[param]:<25} {exp:>10.4f} {actual:>10.4f} {delta:>+10.4f} {status:>8}')

# Stage 5: circular aspect correlation
with open('output/merit_validation/stage5/stage5_results.json') as f:
    s5 = json.load(f)

aspect_circ = s5['corrected_metrics']['aspect']['correlation']
exp_circ = 0.9999
delta_circ = aspect_circ - exp_circ
status_circ = 'PASS' if abs(delta_circ) <= TOLERANCE else 'FAIL'
if status_circ == 'FAIL':
    any_fail = True
print(f'{\"Aspect (circular)\":<25} {exp_circ:>10.4f} {aspect_circ:>10.4f} {delta_circ:>+10.4f} {status_circ:>8}')

print()
print('* Aspect Pearson correlation is low (0.65) due to a known pre-existing')
print('  bug in stage4: it compares degrees vs radians without conversion.')
print('  The true circular correlation is 0.9999 (computed by stage5).')
print()
if any_fail:
    print('RESULT: FAIL — one or more parameters outside tolerance')
    sys.exit(1)
else:
    print('RESULT: PASS — geographic CRS code path unchanged')
    sys.exit(0)
"
