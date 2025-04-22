#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --output=logs/train_classifier_%J.log
#SBATCH --mail-type=NONE                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition gpu
#SBATCH --constraint="h100"
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32gb                 # Job memory request
#SBATCH --time=2-00:00:00                  # Time limit hrs:min:sec
date;hostname;pwd

module purge
module load miniconda
conda activate unsloth_env

python train_accept_classifier.py \
    --model_name google/gemma-3-1b-it \
    --dataset_path /home/ap2853/paper_polish/datasets/iclr2025_papers_cleaned.hf \
    --wandb_project paper_polish \
    --wandb_run_name gemma_3_1b_binary_classifier_cleaned_data \
    --learning_rate 2e-5 \
    --epochs 10 \
    --batch_size 1 \
    --max_length 16384 \
    --output_dir /home/ap2853/scratch/paper_polish/models \
    --bf16 \
    --gradient_accumulation_steps 16