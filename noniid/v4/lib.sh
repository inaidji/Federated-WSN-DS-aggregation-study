#!/bin/bash
# Shared helper: run one federated experiment (server + N clients) in its own directory.
#   run_one <rundir> <server_py> <client_py> <n_clients> <env assignments...>
# Resumable (skips finished runs). If a client dies mid-run (rare TensorFlow fault) the run is
# aborted immediately and retried, up to 3 attempts; a 45-minute watchdog also kills hung runs.
_run_done() { grep -q "\[Round ${FL_NUM_ROUNDS}\] Global metrics" "$1/logs/server.log" 2>/dev/null; }

run_one() {
  local RUNDIR="$1" SRVPY="$2" CLIPY="$3" NC="$4"; shift 4
  mkdir -p "$RUNDIR/logs"
  if _run_done "$RUNDIR"; then echo "[skip] $RUNDIR already done"; return 0; fi
  local attempt
  for attempt in 1 2 3; do
    rm -rf "$RUNDIR/preds"
    find "$RUNDIR" -maxdepth 1 \( -name 'metrics_history_*.json' -o -name 'krum_selection_*.json' \) -delete
    ( cd "$RUNDIR"
      env "$@" python "$SRVPY" > logs/server.log 2>&1 &
      SRV=$!
      ( sleep 2700; kill $SRV 2>/dev/null ) & WD=$!
      sleep 12
      CP=()
      for c in $(seq 1 "$NC"); do
        env "$@" WSN_PART=$c python "$CLIPY" > logs/client${c}.log 2>&1 &
        CP+=($!); sleep 2
      done
      while kill -0 $SRV 2>/dev/null; do
        if ! grep -q "\[Round ${FL_NUM_ROUNDS}\] Global metrics" logs/server.log; then
          for p in "${CP[@]}"; do
            if ! kill -0 $p 2>/dev/null; then echo "[crash] a client died; aborting attempt"; kill $SRV 2>/dev/null; break 2; fi
          done
        fi
        sleep 10
      done
      kill $WD 2>/dev/null; sleep 2
      for p in "${CP[@]}"; do kill $p 2>/dev/null; done
    )
    lsof -ti tcp:${WSN_PORT:-8081} | xargs kill -9 2>/dev/null
    sleep 3
    if _run_done "$RUNDIR"; then return 0; fi
    echo "[retry] $RUNDIR attempt $attempt failed"
  done
  echo "[FAILED] $RUNDIR"
}
