#!/bin/bash
#SBATCH --job-name=paper_polish_finetune_gemma3-4b             # Job name
#SBATCH --output logs/paper_polish_finetune_gemma4b_reviews_%J.log        # Output log file
#SBATCH --partition gpu
#SBATCH --requeue
#SBATCH --nodes=1	
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gpus=1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32gb                                  # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

cd /home/ap2853/paper_polish
module load miniconda
conda activate /home/ap2853/.conda/envs/unsloth_env

# #################################################################
# NOTE: There's a bug in transformers that causes eval to fail when
# using bfloat16 with gemma-3. To get around this, find the model's
# config.json file by running
#   `find /path/to/huggingface/cache -name config.json 2>/dev/null`
# and change use_config to "false" in the file. Change it back
# before inference.
# #################################################################

python train_review_generator.py \
    --model_name google/gemma-3-4b-it \
    --dataset_path /gpfs/radev/home/ap2853/paper_polish/datasets/merged_paper_reviews_2025_individual_samples.hf \
    --output_dir /gpfs/radev/home/ap2853/paper_polish/models \
    --wandb_project paper_polish \
    --wandb_run_name finetune_reviewer_agent_gemma3-4b \
    --learning_rate 2e-5 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_length 16000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --num_train_epochs 1 \
    --max_eval_samples 100 \
    --logging_steps 5 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 3 \
    --seed 42 \