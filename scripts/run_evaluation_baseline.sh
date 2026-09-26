#!/bin/bash

# Script to run run_evaluation_baseline.py with multiple random seeds (all baseline methods)
# Usage: ./run_evaluation_baseline.sh [--sequential | --parallel] [--skip_kg]
# Default: sequential, all datasets (including KG). Use --skip_kg to skip FB15k237/WN18RR.
#
# Options:
#   --sequential    Run seeds one after another (default)
#   --parallel      Run all seeds in parallel (background)
#   --skip_kg       Only homogeneous graphs (Cora, CiteSeer, PubMed); skip FB15k237, WN18RR

# Activate the specified conda environment
source ~/Data/miniconda3/etc/profile.d/conda.sh
conda activate graph_unlearning_gpu


# Set working directory
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

declare -A METHOD_MAP=(
    [1]="retrain"
    [2]="gradient_ascent"
    [3]="gif"
    [4]="gnndelete"
    [5]="grapheraser"
    [6]="full_method"
    [7]="inpo"
    [8]="etr"
    [9]="s_cuno"
)
ALL_METHODS_FULL="retrain,gradient_ascent,gif,gnndelete,grapheraser,full_method,inpo,etr,s_cuno"

echo "Select methods to run:"
echo "  0 = ALL (retrain, gradient_ascent, gif, gnndelete, grapheraser, full_method, inpo, etr, s_cuno)"
echo "  1 = retrain"
echo "  2 = gradient_ascent"
echo "  3 = gif"
echo "  4 = gnndelete"
echo "  5 = grapheraser"
echo "  6 = full_method  [CUNO - default]"
echo "  7 = inpo"
echo "  8 = etr"
echo "  9 = s_cuno       [S-CUNO - self-adaptive curriculum-weighted NPO]"
echo ""
printf "Enter number(s), comma-separated (default 0): "
read -r METHOD_SELECT
METHOD_SELECT="${METHOD_SELECT:-0}"

if [ "$METHOD_SELECT" == "0" ]; then
    ALL_METHODS="$ALL_METHODS_FULL"
else
    ALL_METHODS=""
    IFS=',' read -ra NUMS <<< "$METHOD_SELECT"
    for num in "${NUMS[@]}"; do
        num=$(echo "$num" | tr -d ' ')
        if [ -n "${METHOD_MAP[$num]+x}" ]; then
            [ -n "$ALL_METHODS" ] && ALL_METHODS="${ALL_METHODS},"
            ALL_METHODS="${ALL_METHODS}${METHOD_MAP[$num]}"
        else
            echo "Warning: unknown method number '$num', skipping."
        fi
    done
    if [ -z "$ALL_METHODS" ]; then
        echo "Error: no valid methods selected. Exiting."
        exit 1
    fi
fi

# Define 5 different random seeds
SEEDS=(42 123 456 789 2024)

# Generate timestamp for this batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PID_FILE="logs/evaluation_baseline_batch_${TIMESTAMP}.pids"

# Check run mode (default: sequential) and dataset scope
RUN_MODE="sequential"
# Default: run all datasets including KG (--no_skip_kg overrides script's DEBUG_SKIP_KG)
SKIP_KG_FLAG="--skip_kg"
for arg in "$@"; do
    case "$arg" in
        --parallel)
            RUN_MODE="parallel"
            ;;
        --skip_kg)
            SKIP_KG_FLAG="--skip_kg"
            ;;
        --sequential)
            RUN_MODE="sequential"
            ;;
    esac
done

echo "=========================================="
echo "Starting Baseline Evaluation (All Methods)"
echo "=========================================="
echo "Seeds: ${SEEDS[*]}"
echo "Methods: $ALL_METHODS"
echo "Run Mode: $RUN_MODE"
echo "Skip KG: $([ -n \"$SKIP_KG_FLAG\" ] && echo 'Yes' || echo 'No (all datasets)')"
echo "Timestamp: $TIMESTAMP"
echo ""

# Interactive: load cached unlearned models?
LOAD_CACHE_FLAG="--no_load_cached_unlearned"
printf "Load cached unlearned models when available? (y/n, default n): "
read -r LOAD_CACHE_REPLY
case "$(echo "$LOAD_CACHE_REPLY" | tr '[:upper:]' '[:lower:]')" in
    y|yes|1) LOAD_CACHE_FLAG="--load_cached_unlearned"; echo "✓ Will load cache when available." ;;
    *)       echo "✓ Will not load cache; unlearning will run from scratch." ;;
esac
echo ""

# Clear PID file
> "$PID_FILE"

if [ "$RUN_MODE" == "parallel" ]; then
    # ==================== Parallel Mode ====================
    echo "Running all seeds in PARALLEL..."
    echo ""
    
    for SEED in "${SEEDS[@]}"; do
        LOG_FILE="logs/evaluation_baseline_seed${SEED}_${TIMESTAMP}.log"
        
        echo "Starting seed $SEED..."
        echo "  Log: $LOG_FILE"
        
        # Run in background: all baseline methods, optional skip_kg
        nohup python run_evaluation_baseline.py --seed $SEED --methods "$ALL_METHODS" $SKIP_KG_FLAG $LOAD_CACHE_FLAG > "$LOG_FILE" 2>&1 &
        
        PID=$!
        echo "$SEED:$PID" >> "$PID_FILE"
        echo "  PID: $PID"
        echo ""
    done
    
    echo "=========================================="
    echo "All ${#SEEDS[@]} processes started!"
    echo ""
    echo "To monitor all logs:"
    echo "  tail -f logs/evaluation_baseline_seed*_${TIMESTAMP}.log"
    echo ""
    echo "To check running processes:"
    echo "  cat $PID_FILE"
    echo "  ps -p \$(cut -d: -f2 $PID_FILE | tr '\\n' ',')"
    echo ""
    echo "To stop all processes:"
    echo "  for pid in \$(cut -d: -f2 $PID_FILE); do kill \$pid 2>/dev/null; done"
    echo "=========================================="
    
else
    # ==================== Sequential Mode ====================
    echo "Running seeds SEQUENTIALLY..."
    echo ""
    
    TOTAL_SEEDS=${#SEEDS[@]}
    CURRENT=1
    
    for SEED in "${SEEDS[@]}"; do
        LOG_FILE="logs/evaluation_baseline_seed${SEED}_${TIMESTAMP}.log"
        
        echo "=========================================="
        echo "[$CURRENT/$TOTAL_SEEDS] Running with seed: $SEED"
        echo "=========================================="
        echo "Log file: $LOG_FILE"
        echo ""
        
        # Run and wait for completion: all baseline methods, optional skip_kg
        python run_evaluation_baseline.py --seed $SEED --methods "$ALL_METHODS" $SKIP_KG_FLAG $LOAD_CACHE_FLAG > "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Seed $SEED completed successfully"
        else
            echo "✗ Seed $SEED failed with exit code $EXIT_CODE"
        fi
        echo ""
        
        CURRENT=$((CURRENT + 1))
    done
    
    echo "=========================================="
    echo "All ${#SEEDS[@]} seeds completed!"
    echo ""
    echo "Results are saved in:"
    echo "  - results/baseline_comparison.csv (and baseline_comparison_seed{N}.csv)"
    echo "  - Individual logs in logs/ directory"
    echo ""
    echo "To aggregate results across seeds, check:"
    for SEED in "${SEEDS[@]}"; do
        echo "  - logs/evaluation_baseline_seed${SEED}_${TIMESTAMP}.log"
    done
    echo "=========================================="
fi
