#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Apr 15 2025 10:11:16 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="small_test_1_relax"

# The project ID which this job should run under:
#SBATCH --account="punim0539"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-0:10:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

PDB_ID="1A23"
PDB_PATH="./andrewh/small_test_1/$PDB_ID.pdb"
OUT_DIR="./andrewh/small_test_1/$PDB_ID"

# Create output directory
mkdir -p "$OUT_DIR"

# The job command(s):
./Rosetta/rosetta.binary.linux.release-371/main/source/bin/relax.static.linuxgccrelease -in:file:s "$PDB_PATH" -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false -out:path:all "$OUT_DIR"

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
