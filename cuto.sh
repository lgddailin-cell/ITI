#!/bin/bash
#JSUB -q gpu
#JSUB -m gpu06
#JSUB -e error.%J
#JSUB -o output.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.12-cuda11.3
python /home/25171213997/ITI/codes/run.py
