#!/bin/bash
#SBATCH --job-name=reviewer_inference
#SBATCH --output=logs/reviewer_inference_%J.log
#SBATCH --mail-type=NONE                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition gpu
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # Run on a single CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32gb                 # Job memory request
#SBATCH --time=1:00:00                  # Time limit hrs:min:sec
date;hostname;pwd

module purge
module load miniconda
conda activate unsloth_env

DATASET_PATH="/gpfs/radev/home/ap2853/project/paper_polish_dataset/merged_paper_reviews_2025_individual_samples.hf"

python reviewer_inference.py \
    --model_path /gpfs/radev/home/ap2853/paper_polish/models/checkpoint-1000/ \
    --dataset_path $DATASET_PATH \
    --output_file /gpfs/radev/home/ap2853/paper_polish/results/rank128_ckpt1000.json \
    --max_test_samples 100 \
    --max_tokens 512