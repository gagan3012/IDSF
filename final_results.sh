#!/bin/bash
#SBATCH --time=0:59:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-mageed
#SBATCH --job-name=train_image_v2
#SBATCH --output=../logs/%x.out
#SBATCH --error=../logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8 scipy-stack gcc arrow cuda cudnn

source ~/ENV38_default/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python final_results.py 
