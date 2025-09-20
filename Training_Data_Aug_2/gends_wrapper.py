import subprocess
import sys
import os

def run_script(script_path, args=None):
    """
    Run a Python script with optional command-line arguments.
    """
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Script {script_path} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    # Path to your scripts (adjust if needed)
    script1 = os.path.join(os.path.dirname(__file__), "gends.py")
    script2 = os.path.join(os.path.dirname(__file__), "gends_followup.py")

    # Example: pass through command-line args to both scripts
    args = sys.argv[1:]

    # Run first script
    run_script(script1, args)

    output_arg = None
    for i, a in enumerate(args):
        if a in ("-o", "--output") and i + 1 < len(args):
            output_arg = args[i + 1]
            break

    h5_file = output_arg + ".h5"
    run_script(script2, [h5_file])

    # Remove gends.h5
    if output_arg and os.path.exists(h5_file):
        os.remove(h5_file)
        print(f"Removed temporary file {h5_file}")


if __name__ == "__main__":
    main()