#!/bin/bash
# Clear the runs directory

if [ -d "runs" ]; then
    rm -rf runs/
    echo "Cleared runs/"
else
    echo "No runs/ directory found"
fi
