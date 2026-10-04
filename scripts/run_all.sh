#!/usr/bin/env bash
# Run every case: for each, the base config (GetFEM, whole pipeline up to fem_fit) and then its
# Warp / torch variants (fem_fit only, on the base fem_setup). Cases run in parallel, the three
# fits of one case one after the other. Logs in logs_run_all/<config>.txt.
#
#   bash scripts/run_all.sh                       # all cases
#   bash scripts/run_all.sh karl04 patient_10     # only these
#   CORES="warp torch" bash scripts/run_all.sh    # skip the GetFEM fit (still runs the base steps up to fem_setup)

cd "$(dirname "$0")/.." || exit 1
CASES=${*:-"karl04 karl04_wall patient_2 patient_10"}
CORES=${CORES:-"getfem warp torch"}
mkdir -p logs_run_all
export PYTHONUNBUFFERED=1 CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1

run_case() {
    local c=$1
    if [[ " $CORES " == *" getfem "* ]]; then
        python main.py --config configs/$c.json > logs_run_all/$c.txt 2>&1
    else                                                   # base steps without the GetFEM fit
        steps=$(python -c "import json; print(' '.join(s for s in json.load(open('configs/$c.json'))['steps'] if s != 'fem_fit'))")
        python main.py --config configs/$c.json --steps $steps > logs_run_all/$c.txt 2>&1
    fi
    echo "$(date +%H:%M) $c: exit $?"
    for core in warp torch; do
        [[ " $CORES " == *" $core "* ]] || continue
        python main.py --config configs/${c}_$core.json > logs_run_all/${c}_$core.txt 2>&1
        echo "$(date +%H:%M) ${c}_$core: exit $?"
    done
}

for c in $CASES; do
    run_case "$c" &
done
wait
echo "all done; results in results/, logs in logs_run_all/"
