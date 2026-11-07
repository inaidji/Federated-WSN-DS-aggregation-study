#!/bin/bash
cd "$(dirname "$0")"
./run_main.sh && ./run_poison.sh
echo "ALL DONE v4"
