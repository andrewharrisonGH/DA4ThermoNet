#!/bin/bash

# Partition for the job:
#SBATCH --partition=gpu-a100

# Number of nodes
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_ensemble_train2"

# The project ID which this job should run under:
#SBATCH --account="punim0539"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# The amount of memory in megabytes per node:
#SBATCH --mem=32768

# Use this email address:
#SBATCH --mail-user=andrew.harrison@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=2-00:00:00

#SBATCH --array=1-10
#SBATCH --output=logs/ens_%A_%a.out
#SBATCH --error=logs/ens_%A_%a.err

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# The modules to load:
module load foss/2022a Python/3.10.4 TensorFlow/2.11.0-CUDA-11.7.0

# Run the job from the directory where it was launched (default)

# The job command(s):
# Run ensemble training (each task trains one member)
MEMBER_IDX=$SLURM_ARRAY_TASK_ID

srun python train_ensemble2.py \
    --direct_features Q1744_tensorsi36_fwd.npy \
    --inverse_features Q1744_tensorsi36_rev.npy \
    --direct_targets Q1744_tensorsi36_fwd_ddg.txt \
    --epochs 200 \
    --prefix DAi36rotTN \
    --member $MEMBER_IDX \
    --k 10
    

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s