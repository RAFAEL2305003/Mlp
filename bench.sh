#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODELS_DIR="$ROOT/models"
LOGS_DIR="$ROOT/bench_logs"
RUNS=5

mkdir -p "$LOGS_DIR"

icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 src/nn.cpp -o nn

shopt -s nullglob
for model in "$MODELS_DIR"/*.json; do
    name="$(basename "$model" .json)"
    log="$LOGS_DIR/${name}.log"
    : > "$log"

    echo "=== $name ===" | tee -a "$log"
    total=0
    for run in $(seq 1 "$RUNS"); do
        start=$(date +%s.%N)
        ./nn "$model" > /dev/null
        end=$(date +%s.%N)
        elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
        total=$(awk -v t="$total" -v x="$elapsed" 'BEGIN { printf "%.3f", t + x }')
        echo "run $run: ${elapsed}s" | tee -a "$log"
    done
    avg=$(awk -v t="$total" -v n="$RUNS" 'BEGIN { printf "%.3f", t / n }')
    echo "total: ${total}s  avg: ${avg}s" | tee -a "$log"
done
