import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    print(f"\n=== Running: {description} ===")
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # 1. Split training data
    run_command(
        [sys.executable, str(script_dir / "split_data.py")],
        "Splitting training data into local train/validation sets"
    )
    
    # 2. Run baselines on the local validation split
    run_command(
        [
            sys.executable,
            str(project_root / "res" / "provided_res" / "baseline" / "run_baselines.py"),
            "--train", "res/dataset/local_train.csv",
            "--forecast-index", "res/dataset/local_forecast_index_validation.csv",
            "--output-dir", "output/local_baselines"
        ],
        "Generating baseline forecasts on local validation split"
    )
    
    # 3. Evaluate baseline predictions
    run_command(
        [sys.executable, str(script_dir / "evaluate_predictions.py")],
        "Evaluating forecasts and precomputing model metrics"
    )
    
    # 4. Start dashboard
    print("\n=== Starting Evaluation Dashboard ===")
    dashboard_path = script_dir / "dashboard.py"
    print(f"Executing: {sys.executable} -m streamlit run {dashboard_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

if __name__ == "__main__":
    main()
