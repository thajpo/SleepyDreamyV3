#!/bin/bash
# Clear temporary/generated files from repo

rm -rf runs/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -f *.pt
rm -rf .pytest_cache/

echo "Cleaned: runs/, __pycache__, *.pt, .pytest_cache/"
