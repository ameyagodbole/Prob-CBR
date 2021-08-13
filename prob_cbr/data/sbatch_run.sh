#!/usr/bin/env bash

#SBATCH --job-name=path-collection
#SBATCH --output=/mnt/nfs/scratch1/rajarshi/deep_case_based_reasoning/get_paths_parallel-%A_%a.out
#SBATCH --partition=longq
#SBATCH --time=07-00:00:00
#SBATCH --mem=40G
#SBATCH --array=0-99

# Set to scratch/work since server syncing will occur from here
# Ensure sufficient space else runs crash without error message

# Flag --count specifies number of hyperparam settings (runs) tried per job
# If SBATCH --array flag is say 0-7 (8 jobs) then total (8 x count)
# hyperparam settings will be tried

wandb agent rajarshd/Prob-CBR-prob_cbr_data/v2isk3gn