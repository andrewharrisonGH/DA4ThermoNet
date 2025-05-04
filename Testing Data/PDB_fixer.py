import os
import subprocess

input_path_pdb = r"C:\Users\andyh\OneDrive\Documents\ResearchProject\Experiments\DA4ThermoNet\Testing Data\S2648\pdbs\1LVE.pdb"
output_path_pdb = r"C:\Users\andyh\OneDrive\Documents\ResearchProject\Experiments\DA4ThermoNet\Testing Data\S2648\pdbs\1LVE_new.pdb"

if input_path_pdb.endswith(".pdb"):
    subprocess.run(["python", r"C:\Users\andyh\OneDrive\Documents\ResearchProject\Experiments\DA4ThermoNet\PDBrenum_git\PDBrenum.py", input_path_pdb, "-o", output_path_pdb])
