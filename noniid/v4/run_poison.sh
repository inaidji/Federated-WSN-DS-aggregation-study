#!/bin/bash
# v4 poisoning study: K=5 partition, attackers IGNORE the proximal term, boosted sign-flip (lambda=3),
# 5 aggregators x 5 conditions x 3 seeds. Krum's per-round selection is logged by the server.
cd "$(dirname "$0")"; V4=$(pwd); source "$V4/lib.sh"
export FL_NUM_ROUNDS=40 WSN_LOCAL_EPOCHS=1 WSN_LIGHT=1 WSN_VAL_FRAC=0.1 WSN_DUMP_LAST=5 WSN_MIN_CLIENTS=5
export WSN_ATTACKER_IGNORES_MU=1 WSN_ATTACK_TRAINABLE_ONLY=1 WSN_POISON_LAMBDA=3.0
SEEDS=${SEEDS:-"0 1 2"}
# cond | poison env | krum f
CONDS=(
  "clean|WSN_POISON=none|1"
  "lf1|WSN_POISON=labelflip WSN_POISON_PARTS=3|1"
  "lf2|WSN_POISON=labelflip WSN_POISON_PARTS=3,4|2"
  "sf1|WSN_POISON=signflip WSN_POISON_PARTS=3|1"
  "sf2|WSN_POISON=signflip WSN_POISON_PARTS=3,4|2"
)
STRATS=("fedavg|" "fedprox|WSN_MU=1.0" "fedtrimmedavg|WSN_BETA=0.2" "krum|" "fedmedian|")
N=$(( ${#CONDS[@]} * ${#STRATS[@]} * $(echo $SEEDS | wc -w) )); i=0
for SEED in $SEEDS; do for cond in "${CONDS[@]}"; do for st in "${STRATS[@]}"; do
  IFS='|' read -r COND POISON KF <<< "$cond"; IFS='|' read -r STRAT EXTRA <<< "$st"; i=$((i+1))
  TAG="s${SEED}_${COND}_${STRAT}"
  echo "[POISON $i/$N] $TAG  $(date +%H:%M)"
  KR=""; [ "$STRAT" = "krum" ] && KR="WSN_KRUM_F=$KF"
  run_one "$V4/poison/$TAG" "$V4/../robust/_server.py" "$V4/../robust/_client.py" 5 \
      WSN_STRATEGY=$STRAT WSN_STRATEGY_TAG=$TAG WSN_SEED=$SEED $POISON $EXTRA $KR
done; done; done
echo "POISON DONE"
