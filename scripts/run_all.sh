#!/usr/bin/env bash
# Run cases: configs/<case>_torch.json (whole pipeline, torch core), cases in parallel.
# Logs in logs_run_all/<case>.txt. Extra arguments after "--" go to main.py.
#
#   bash scripts/run_all.sh                                      # every configs/*_torch.json
#   bash scripts/run_all.sh karl04 karl04_wall                   # only these
#   bash scripts/run_all.sh karl04 karl04_wall -- --steps fem_setup fem_fit   # only the FEM steps

cd "$(dirname "$0")/.." || exit 1
CASES=() EXTRA=()
while [[ $# -gt 0 && $1 != "--" ]]; do CASES+=("$1"); shift; done
[[ $1 == "--" ]] && { shift; EXTRA=("$@"); }
if [[ ${#CASES[@]} -eq 0 ]]; then
    for f in configs/*_torch.json; do c=$(basename "$f" _torch.json); CASES+=("$c"); done
fi
mkdir -p logs_run_all
export PYTHONUNBUFFERED=1 CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1

run_case() {
    local c=$1 rc
    python main.py --config configs/${c}_torch.json "${EXTRA[@]}" > logs_run_all/$c.txt 2>&1
    rc=$?                                                  # before $(date) overwrites it
    echo "$(date +%H:%M) $c: exit $rc"
}

for c in "${CASES[@]}"; do
    run_case "$c" &
done
wait
echo "all done; results in results/, logs in logs_run_all/"
