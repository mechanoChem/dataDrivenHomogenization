#!/bin/bash

#SBATCH -J zxx_gpu1
#SBATCH -p gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=25GB
#SBATCH -t 47:00:00 

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xxzh@umich.edu

hostname
root=/home/xxzh/auto_test/mylib
export PYTHONPATH=PYTHONPATH:$HOME/python_projects/

module purge
#module load gnu/4.9.2 cmake/3.9.1
module load cuda/9.2
#module load cuda/8.0
conda init
conda activate tf-gpu
nvcc --version
timeout 2 nvidia-smi

python hyper_parameter_search.py

##########SBATCH -A TG-MSS160003
