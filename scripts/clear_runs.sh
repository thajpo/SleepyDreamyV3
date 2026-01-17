#!/bin/bash
# Clear the runs directory
# Usage: ./clear_runs.sh [--under STEPS]
#   --under STEPS  Only delete runs with fewer than STEPS steps

UNDER_STEPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --under)
            UNDER_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--under STEPS]"
            exit 1
            ;;
    esac
done

if [ ! -d "runs" ]; then
    echo "No runs/ directory found"
    exit 0
fi

if [ -z "$UNDER_STEPS" ]; then
    # Clear all runs
    rm -rf runs/
    echo "Cleared all runs/"
else
    # Clear only runs under specified step count
    deleted=0
    skipped=0
    for run_dir in runs/*/; do
        [ -d "$run_dir" ] || continue

        # Skip sweep directories (they contain nested runs)
        if [[ "$run_dir" == *"sweeps"* ]]; then
            echo "Skipping sweep directory: $run_dir"
            ((skipped++))
            continue
        fi

        # Must have a checkpoints directory to evaluate
        if [ ! -d "${run_dir}checkpoints" ]; then
            echo "Skipping (no checkpoints dir): $run_dir"
            ((skipped++))
            continue
        fi

        max_step=0
        has_checkpoint=false

        # Check checkpoint_step_*.pt files
        shopt -s nullglob
        for ckpt in "$run_dir"checkpoints/checkpoint_step_*.pt; do
            has_checkpoint=true
            step=$(basename "$ckpt" | sed 's/checkpoint_step_\([0-9]*\)\.pt/\1/')
            [ "$step" -gt "$max_step" ] 2>/dev/null && max_step=$step
        done
        shopt -u nullglob

        # Check for final checkpoint - read max_steps from config
        if [ -f "${run_dir}checkpoints/checkpoint_final.pt" ]; then
            has_checkpoint=true
            if [ -f "${run_dir}config.json" ]; then
                cfg_steps=$(grep -o '"max_steps": [0-9]*' "${run_dir}config.json" 2>/dev/null | grep -o '[0-9]*' | head -1)
                if [ -n "$cfg_steps" ] && [ "$cfg_steps" -gt "$max_step" ]; then
                    max_step=$cfg_steps
                fi
            fi
        fi

        # Skip if no checkpoints found (can't determine step count)
        if [ "$has_checkpoint" = false ]; then
            echo "Skipping (no checkpoints): $run_dir"
            ((skipped++))
            continue
        fi

        if [ "$max_step" -lt "$UNDER_STEPS" ]; then
            rm -rf "$run_dir"
            echo "Deleted: $run_dir (max_step=$max_step)"
            ((deleted++))
        else
            echo "Kept: $run_dir (max_step=$max_step)"
        fi
    done
    echo "Deleted $deleted runs under $UNDER_STEPS steps, skipped $skipped"
fi
