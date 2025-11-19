#!/bin/bash

# Run script for ML Inference Pipeline
# This script will be executed on each node

echo "Starting pipeline on Node $NODE_NUMBER..."
# python3 pipeline.py

# Memory profile output file (unique per node)
MEM_PROFILE="mem_profile_node_${NODE_NUMBER}.dat"
MEM_PLOT="mem_plot_node_${NODE_NUMBER}.png"


# Function to generate plot
generate_plot() {
    echo "Generating memory plot for Node $NODE_NUMBER..."
    python3 generate_plot.py "$MEM_PROFILE" "$MEM_PLOT"
}

# Trap Ctrl+C to generate plot before exiting
trap 'echo "Ctrl+C pressed!"; generate_plot; exit 1' SIGINT

# Run the pipeline with memory profiling
# mprof run records memory usage over time
mprof run --output "$MEM_PROFILE" python3 pipeline.py

# Generate plot after the pipeline exits naturally
generate_plot
