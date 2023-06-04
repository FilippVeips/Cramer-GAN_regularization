#!/bin/bash
#SBATCH --job-name="faveyps_RICH_GAN_KDE"
#SBATCH --time=3-00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=1

#SBATCH --error=faveyps_RICH_GAN_KDE-%j.err        # Файл для вывода ошибок 
#SBATCH --output=faveyps_RICH_GAN_KDE-%j.log       # Файл для вывода результатов 

module purge
module load Python/Anaconda_v11.2021

conda init bash
conda deactivate
conda activate mokhnenko_env # заменить на свой GAN_env 

python3 run_single_job.py --config="./richgan/configs/simple.mc/simple.mc.muon.config.yaml" --schema=training --no_uuid_suffix