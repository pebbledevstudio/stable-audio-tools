#!/bin/bash

# Base directory containing the checkpoints
base_dir=~/stable-audio-tools/checkpoints/lightning_logs/lbiqwe4k/checkpoints

# Iterate over each directory in the base directory
for dir in "$base_dir"/*/; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Extract the directory name
        dir_name=$(basename "$dir")
        
        # Create the output directory if it doesn't exist
        mkdir -p "./fp32/$dir_name"
        
        # Run the python script with the directory path
        python zero_to_fp32.py "$dir" "./fp32/$dir_name"
    fi
done