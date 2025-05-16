#!/bin/bash
# Put your slurm commands here

accelerate launch --num_processes=1 {base_path}/main.py --batch_size {batch_size} --model_name {model_name} --split_name {split_name} --dataset_name {dataset_name} --output_dir {model_name}_results --n_take {n_take}
python {base_path}/evaluations/evaluate_images.py \
    --original_dir {model_name}_results/original_images \
    --reconstructed_dir {model_name}_results/reconstructed_images/ \
    --metrics {metrics} \
    --batch_size 16 \
    --num_workers 8 | tee result.txt
