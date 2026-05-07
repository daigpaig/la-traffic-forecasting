#!/bin/bash
# Monitor spatial autonomous research progress

echo "=== SPATIAL OPTIMIZATION MONITOR ==="
echo ""

# Count completed iterations
ITER_COUNT=$(ls -1 logs/spatial_iter_*.txt 2>/dev/null | wc -l)
echo "Completed iterations: $ITER_COUNT"
echo ""

# Show best result so far
echo "=== BEST RESULT SO FAR ==="
if [ $ITER_COUNT -gt 0 ]; then
    best_rmse=999
    best_file=""

    for file in logs/spatial_iter_*.txt; do
        rmse=$(grep "test_rmse=" "$file" | cut -d'=' -f2)
        if (( $(echo "$rmse < $best_rmse" | bc -l) )); then
            best_rmse=$rmse
            best_file=$file
        fi
    done

    echo "File: $(basename $best_file)"
    cat "$best_file"
    echo ""

    # Compare to temporal baseline
    temporal_rmse=15.000
    diff=$(echo "$best_rmse - $temporal_rmse" | bc)
    echo "Comparison to Temporal Baseline (RMSE 15.000):"
    if (( $(echo "$best_rmse < $temporal_rmse" | bc -l) )); then
        echo "✅ BEATING TEMPORAL by $diff"
    else
        echo "❌ Still behind by $diff"
    fi
else
    echo "No results yet. Experiments still running..."
fi

echo ""
echo "=== ALL RESULTS (sorted by RMSE) ==="
for file in logs/spatial_iter_*.txt; do
    if [ -f "$file" ]; then
        rmse=$(grep "test_rmse=" "$file" | cut -d'=' -f2)
        k=$(grep "k_neighbors=" "$file" | cut -d'=' -f2)
        adj=$(grep "adj_type=" "$file" | cut -d'=' -f2)
        layers=$(grep "graph_layers=" "$file" | cut -d'=' -f2)
        echo "$(basename $file): RMSE=$rmse | K=$k | adj=$adj | layers=$layers"
    fi
done | sort -t'=' -k2 -n

echo ""
echo "=== REFRESH ==="
echo "Run this script again to see updated results:"
echo "./experiments/spatio_temporal/monitor.sh"
