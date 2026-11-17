#!/usr/bin/env bash
# Run cases: configs/<case>.json (the steps of the config), cases in parallel.
# Logs in logs_run_all/<case>.txt. Extra arguments after "--" go to main_fit.py.
#
#   bash scripts/run_all.sh                                      # every configs/*.json
#   bash scripts/run_all.sh karl04 karl20                        # only these
#   bash scripts/run_all.sh karl04 karl20 -- --steps fem_setup fit   # only these steps
#   LOG=gE1000 bash scripts/run_all.sh ... -- --set physics.E_Pa=1000 run_name=gE1000   # logs <case>_gE1000.txt
#   JOBS=4 bash scripts/run_all.sh ...                           # at most 4 cases at a time (default: all)
#
# Many cases at once share one GPU and the CPUs: with 28 the solves were ~15x slower, and the
# max_solve_s limit then fails solves that converge in seconds when run alone.

cd "$(dirname "$0")/.." || exit 1
CASES=() EXTRA=()
while [[ $# -gt 0 && $1 != "--" ]]; do CASES+=("$1"); shift; done
[[ $1 == "--" ]] && { shift; EXTRA=("$@"); }
if [[ ${#CASES[@]} -eq 0 ]]; then
    for f in configs/*.json; do c=$(basename "$f" .json); CASES+=("$c"); done
fi
mkdir -p logs_run_all
export PYTHONUNBUFFERED=1 CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1

run_case() {
    local c=$1 rc
    python main_fit.py --config configs/${c}.json "${EXTRA[@]}" > logs_run_all/$c${LOG:+_$LOG}.txt 2>&1
    rc=$?                                                  # before $(date) overwrites it
    echo "$(date +%H:%M) $c: exit $rc"
}

JOBS=${JOBS:-0}
for c in "${CASES[@]}"; do
    if [[ $JOBS -gt 0 ]]; then
        while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do wait -n; done
    fi
    run_case "$c" &
done
wait
echo "all done; results in results/, logs in logs_run_all/"
