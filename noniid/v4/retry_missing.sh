#!/bin/bash
# Waits for the first sweep to end, then re-runs (resumably) only the poisoning runs that did not finish.
cd "$(dirname "$0")"
while ! grep -q "ALL DONE v4" "$1"; do sleep 60; done
./run_poison.sh
echo "RETRY DONE"
