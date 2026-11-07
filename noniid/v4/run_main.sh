#!/bin/bash
# v4 main grid: K=3 highly non-IID partition, 11 configs x 5 seeds, new train/val/test split,
# per-client val/test probabilities of the global model dumped for the last 5 rounds.
cd "$(dirname "$0")"; V4=$(pwd); source "$V4/lib.sh"
export FL_NUM_ROUNDS=40 WSN_LOCAL_EPOCHS=1 WSN_LIGHT=1 WSN_VAL_FRAC=0.1 WSN_DUMP_LAST=5
SEEDS=${SEEDS:-"0 1 2 3 4"}
RUNS=(
  "fedavg|fedavg|"
  "fedavgm|fedavgm|"
  "fedmedian|fedmedian|"
  "fedprox_mu0.1|fedprox|WSN_MU=0.1"
  "fedprox_mu0.5|fedprox|WSN_MU=0.5"
  "fedprox_mu1.0|fedprox|WSN_MU=1.0"
  "fedadam_eta1e-3|fedadam|WSN_ETA=1e-3 WSN_TAU=1e-2"
  "fedadam_eta3e-3|fedadam|WSN_ETA=3e-3 WSN_TAU=1e-2"
  "fedyogi_eta1e-3|fedyogi|WSN_ETA=1e-3 WSN_TAU=1e-2"
  "fedyogi_eta3e-3|fedyogi|WSN_ETA=3e-3 WSN_TAU=1e-2"
  "fedtrimmedavg|fedtrimmedavg|WSN_BETA=0.2"
)
N=$(( ${#RUNS[@]} * $(echo $SEEDS | wc -w) )); i=0
for SEED in $SEEDS; do for entry in "${RUNS[@]}"; do
  IFS='|' read -r TAG STRAT EXTRA <<< "$entry"; i=$((i+1))
  echo "[MAIN $i/$N] s${SEED}_${TAG}  $(date +%H:%M)"
  run_one "$V4/main/s${SEED}_${TAG}" "$V4/../v2/_server.py" "$V4/../v2/_client.py" 3 \
      WSN_STRATEGY=$STRAT WSN_STRATEGY_TAG=s${SEED}_${TAG} WSN_SEED=$SEED $EXTRA
done; done
echo "MAIN DONE"
