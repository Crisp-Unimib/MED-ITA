#!/bin/bash

set -e

if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Exiting."
    exit 1
fi

echo "Running evaluation script..."
python run_eval.py --config-name config.yaml