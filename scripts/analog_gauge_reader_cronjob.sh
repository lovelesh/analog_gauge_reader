#!/bin/bash

# Example shell script
time_start=`date +%Y_%m_%d_%H:%M`
echo "Begin script $time_start"

# Sourcing conda
source ~/miniconda3/etc/profile.d/conda.sh

# activating your particular environment
# may need to give full path, not just the name
conda activate coral

# if you want to check environment
python --version

# you may need to change the directory at this point
cd ~/analog_gauge_reader
echo "Current Directory is set to $PWD"

# run your python script
log_file="log_$time_start.txt"
python pipeline_with_webcam_onnx_runtime.py --input 2 --run > $PWD/logs/$log_file 2>&1
