#!/bin/bash

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_ensemble_train"

# The project ID which this job should run under:
#SBATCH --account="punim0539"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

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

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# The modules to load:
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate tensorflow112

# Run the job from the directory where it was launched (default)

# The job command(s):
python train_ensemble.py --direct_features Q1744_tensors0_fwd.npy --inverse_features Q1744_tensors0_rev.npy --direct_targets Q1744_tensors0_fwd_ddg.txt --epochs 200 --prefix DA0rotTN

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s