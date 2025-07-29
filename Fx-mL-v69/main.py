import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_script(script_name, output_file):
    """Run a Python script and capture its output."""
    print(f"\nRunning {script_name}...")
    output_file.write(f"\n{'='*50}\n")
    output_file.write(f"Running {script_name} at {datetime.now()}\n")
    output_file.write(f"{'='*50}\n\n")
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy()  # Pass current environment variables
        )
        
        # Write output to file
        output_file.write(result.stdout)
        if result.stderr:
            output_file.write("\nErrors:\n")
            output_file.write(result.stderr)
            
        return True
    except subprocess.CalledProcessError as e:
        output_file.write(f"\nError running {script_name}:\n")
        output_file.write(e.stdout)
        output_file.write(e.stderr)
        return False

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Set RUN_ID environment variable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = f"models/run_{timestamp}"
    os.environ["RUN_ID"] = run_id
    
    # Create run directory
    run_dir = Path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Open output file
    output_path = output_dir / "output.txt"
    with open(output_path, "w") as f:
        # Write header
        f.write(f"Pipeline Run Started at {datetime.now()}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"{'='*50}\n\n")
        
        # Run each script in sequence
        scripts = [
            "prepare.py",
            "label.py",
            "train_base.py",
            "train_meta.py",
            "sltp.py",
            "train_ranker.py",
            "simulate.py",
        ]
        
        success = True
        for script in scripts:
            if not run_script(script, f):
                success = False
                f.write(f"\nPipeline failed at {script}\n")
                break
        
        # Write summary
        f.write(f"\n{'='*50}\n")
        f.write("Pipeline Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Status: {'Success' if success else 'Failed'}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Completed at: {datetime.now()}\n")
        
    print(f"\nPipeline {'completed successfully' if success else 'failed'}")
    print(f"Run ID: {run_id}")
    print(f"Output written to {output_path}")

if __name__ == "__main__":
    main()
