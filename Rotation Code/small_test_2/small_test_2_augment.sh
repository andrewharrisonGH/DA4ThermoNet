#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Apr 15 2025 10:11:16 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=sapphire

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_st2_augment"

# The project ID which this job should run under:
#SBATCH --account="punim0539"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-0:30:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# The modules to load:
module load foss/2022a 
module load Python/3.10.4
module load SciPy-bundle/2022.05
module load Biopython/1.79

# Run the job from the directory where it was launched (default)

# The job command(s):
PDB_ID="$1"
INPUT_PDB="./PDBs/$PDB_ID.pdb"
VARIANT_LIST="./Mutations/${PDB_ID}_mutants.csv"
OUT_DIR="./PDB_relaxed/$PDB_ID"

# Create output directory
mkdir -p "$OUT_DIR"

# Step 1 - Relax stating structure
/data/gpfs/projects/punim0539/Rosetta/rosetta.binary.linux.release-371/main/source/bin/relax.static.linuxgccrelease -in:file:s "$INPUT_PDB" -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false -out:path:all "$OUT_DIR"

# Step 2 - Simulate variants
python ./rosetta_relax.py --rosetta-bin /data/gpfs/projects/punim0539/Rosetta/rosetta.binary.linux.release-371/main/source/bin/relax.static.linuxgccrelease -l "$VARIANT_LIST" --base-dir ./PDB_relaxed

# Step 3 - Rotate
python ./rotation_aug.py --pdb_id "$PDB_ID"

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s