#!/bin/bash
#SBATCH --job-name=paper_polish_finetune_gemma3-4b             # Job name
#SBATCH --output training_logs/paper_polish_finetune_gemma4b_reviews_%J.log        # Output log file
#SBATCH --mail-type=ALL                                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=david.jeong@yale.edu                  # Where to send mail
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=2
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb                                   # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

cd /gpfs/radev/home/tj372/project/paper_polish
module load miniconda
conda activate cpsc477-paper-polish

export OMP_NUM_THREADS=8
export NUM_PROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=$NUM_PROC_PER_NODE --nnodes=1 train_review_generator.py \
    --model_name google/gemma-3-4b-it \
    --dataset_path /gpfs/radev/home/tj372/project/paper_polish/merged_paper_reviews_2025 \
    --output_dir /gpfs/radev/home/tj372/scratch/paper_polish/review_models \
    --max_length 12000 \
    --wandb_project paper_polish \
    --wandb_run_name paper_polish_finetune_gemma3-4b \
