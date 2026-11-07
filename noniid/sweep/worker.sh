#!/bin/bash
# usage: worker.sh JOBFILE WORKER_ID NUM_WORKERS ; runs jobs i where i % NUM_WORKERS == WORKER_ID on its own port
cd "$(dirname "$0")"; HERE=$(pwd); source "$HERE/../v4/lib.sh"
JOBS=$1; W=$2; NW=$3
export FL_NUM_ROUNDS=40 WSN_LOCAL_EPOCHS=1 WSN_LIGHT=1 WSN_VAL_FRAC=0.1 WSN_DUMP_LAST=5 WSN_PORT=${4:-$((8100 + W))}
i=0; n=$(wc -l < "$JOBS")
while IFS='|' read -r RUNDIR NC ENVS; do
  if [ $((i % NW)) -eq "$W" ]; then
    echo "[w$W $((i+1))/$n] $(basename $RUNDIR) $(date +%H:%M)"
    run_one "$RUNDIR" "$HERE/_server.py" "$HERE/_client.py" "$NC" $ENVS
  fi
  i=$((i+1))
done < "$JOBS"
echo "[w$W] DONE $(date +%H:%M)"
