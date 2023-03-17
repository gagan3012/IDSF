#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-mageed
#SBATCH --job-name=train_image_v2
#SBATCH --output=/lustre07/scratch/gagan30/arocr/logs/%x.out
#SBATCH --error=/lustre07/scratch/gagan30/arocr/logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8 scipy-stack gcc arrow cuda cudnn

source ~/ENV38_default/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

model_name=$1
column=$2
dataset=$3

accelerate launch --config_file /lustre07/scratch/gagan30/arocr/code/JointIDSF/mt0/accelerate_ds_zero3_cpu_offload_config.yaml \
    peft_lora_seq2seq_accelerate_ds_zero3_offload.py --model_name $model_name \
    --dataset_name $dataset \
    --batch_size 64 \
    --num_epochs 10 \
    --label_column $column \
    --do_test \

# ./batch_train_v3.sh mt0-small intents en