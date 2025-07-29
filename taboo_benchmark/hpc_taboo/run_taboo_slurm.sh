#!/bin/bash


#SBATCH --job-name=taboo_game_local       
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1               
#SBATCH --cpus-per-task=32               
#SBATCH --mem=100GB                        
#SBATCH --time=08:00:00                   
#SBATCH --partition=gpu                   
#SBATCH --gres=gpu:2                      


#SBATCH --output=/mnt/scratch/%u/taboo_game_results/slurm-%j.out
#SBATCH --error=/mnt/scratch/%u/taboo_game_results/slurm-%j.err

echo "Starting job on $(hostname) at $(date)"

module load miniforge/24.7.1 
module load cuda/12.6.2 
module load pytorch/2.5.1 

conda activate taboo-game-env

cd $HOME/mscproj

python run_local_taboo_experiment.py --num-games 100 --use-recommended --max-workers 4

echo "Job finished at: $(date)"