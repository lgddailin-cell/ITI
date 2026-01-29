#!/bin/bash
#JSUB -q gpu
#JSUB -m gpu03
#JSUB -n 10
#JSUB -e error.%J

source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.12-cuda11.3
python run.py --model DistMult --house_dim 2 --house_num 2 --types_num 2 --housd_num 2 --batch_size 200 --negative_sample_size 300 --save_path '/home/25171213997/ITI/FB15k/DistMult/' --valid_steps 10000 --test_log_steps 1000 --save_checkpoint_steps 20000 --hidden_dim 1000 --test_batch_size 4 --data_path '/home/25171213997/ITI/data/FB15k/'
