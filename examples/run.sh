#!/bin/bash

dataset_name_list=("task1-imagenet" "task1-high-resolution" "task1-varying-resolution" "task2-detail-preservation" "task3-movie-posters" "task3-arxiv-abstracts" "task3-multilingual_Chinese" "task3-multilingual_Hindi" "task3-multilingual_Japanese" "task3-multilingual_Korean")
model_name_list=("chameleon" "llamagen-ds16" "llamagen-ds8" "flowmo_lo" "flowmo_hi" "open_magvit2" "titok_l32" "titok_b64" "titok_s128" "titok_bl64" "titok_bl128" "titok_sl256" "janus_pro_1b" "maskbit_18bit" "maskbit_16bit" "var_256" "var_512" "SD3.5L" "gpt4o" "llamagen-ds16-t2i" "infinity_d32" "infinity_d64" "bsqvit" "FLUX.1-dev")

batch_size=1

if command -v sbatch >/dev/null 2>&1; then
  has_slurm=true
else
  has_slurm=false
fi

shell_dir=$(cd "$(dirname "$0")";pwd)
echo "shell_dir: ${shell_dir}"
base_path="${shell_dir}/../"

for dataset_name in "${dataset_name_list[@]}"
do
  cd ${shell_dir}
  folder_dir="${dataset_name}"
  mkdir ${folder_dir}

  metrics="fid ssim psnr lpips"
  split_name="test"
  n_take=-1

  if [[ $dataset_name == task3-multilingual_* ]]; then
    split_name="${dataset_name##*_}"
    dataset_name="${dataset_name%_*}"
  fi
  if [ "$dataset_name" = "task1-imagenet" ]; then
    split_name="val"
  fi

  if [ "$dataset_name" = "task1-varying-resolution" ]; then
    batch_size=1
  fi
  if [ "$dataset_name" = "task3-movie-posters" ]; then
    metrics="fid ssim psnr lpips cer wer"
  fi
  if [ "$dataset_name" = "task3-arxiv-abstracts" ]; then
    metrics="fid ssim psnr lpips cer wer"
  fi
  if [ "$dataset_name" = "task3-multilingual" ]; then
    metrics="fid ssim psnr lpips cer"
  fi

  for model_name in "${model_name_list[@]}"
  do
    if [ "$dataset_name" = "task1-imagenet" ] && [ "$model_name" = "gpt4o" ]; then
      n_take=100
    fi
    cd ${shell_dir}

    work_dir="${folder_dir}/${model_name}"
    echo "model_name: ${model_name}, work_dir: ${work_dir}"
    mkdir ${work_dir}

    cp submit.sh ${work_dir}
  
    cd ${work_dir}
    sed -i "s|{model_name}|${model_name}|g" submit.sh
    sed -i "s|{split_name}|${split_name}|g" submit.sh
    sed -i "s|{dataset_name}|${dataset_name}|g" submit.sh
    sed -i "s|{batch_size}|${batch_size}|g" submit.sh
    sed -i "s|{base_path}|${base_path}|g" submit.sh
    sed -i "s|{metrics}|${metrics}|g" submit.sh
    sed -i "s|{n_take}|${n_take}|g" submit.sh
  
#     if [ "$has_slurm" = true ]; then
#       res=$(sbatch ./submit.sh)
#       res=($res)
#       task_id=${res[-1]}
#       echo "task_id: ${task_id}"
#       touch "task_id_${task_id}"
#     else
#       echo "Slurm not detected, running with bash..."
#       bash ./submit.sh
#     fi

    bash ./submit.sh

  done
done
