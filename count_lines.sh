#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR="$1"

total_lines=$(find "$DIR" \
    \( -name "*.cpp" -o -name "*.h" \) \
    ! -name "rapidcsv.h" \
    -type f -print0 | \
    xargs -0 cat | wc -l)

echo "Total lines (excluding rapidcsv.h): $total_lines"
