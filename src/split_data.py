import argparse
from pathlib import Path
import pandas as pd

def main():
    # Resolve project paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "res" / "dataset"
    
    train_path = dataset_dir / "train.csv"
    
    if not train_path.exists():
        print(f"Error: Could not find training data at {train_path}")
        print("Please ensure the dataset is downloaded first using: uv run src/download_data.py")
        return

    print(f"Loading dataset from {train_path}...")
    df = pd.read_csv(train_path)
    
    print("Sorting dataset by series_id and timestamp...")
    df = df.sort_values(by=["series_id", "timestamp"]).reset_index(drop=True)
    
    # 336 steps is the validation horizon
    val_horizon = 336
    print(f"Splitting dataset (holding out last {val_horizon} hours per series)...")
    
    # Get last val_horizon steps for validation
    val_df = df.groupby("series_id", as_index=False).tail(val_horizon)
    
    # Preceding steps go to training
    train_df = df.drop(val_df.index).reset_index(drop=True)
    
    # Re-reset index for validation
    val_df = val_df.reset_index(drop=True)
    
    print(f"Split results:")
    print(f" - Train samples: {len(train_df)} (approx. {len(train_df) // 96} hours per series)")
    print(f" - Validation samples: {len(val_df)} ({val_horizon} hours per series)")
    
    # Write training set
    local_train_path = dataset_dir / "local_train.csv"
    print(f"Saving training split to {local_train_path}...")
    train_df.to_csv(local_train_path, index=False)
    
    # Write validation input (without target column)
    local_val_input_path = dataset_dir / "local_validation_input.csv"
    print(f"Saving validation input split to {local_val_input_path}...")
    val_df.drop(columns=["target"]).to_csv(local_val_input_path, index=False)
    
    # Write validation targets (ground truth)
    local_val_targets_path = dataset_dir / "local_validation_targets.csv"
    print(f"Saving validation targets (ground truth) to {local_val_targets_path}...")
    val_df[["series_id", "timestamp", "target"]].to_csv(local_val_targets_path, index=False)
    
    # Write validation forecast index
    local_val_index_path = dataset_dir / "local_forecast_index_validation.csv"
    print(f"Saving validation forecast index to {local_val_index_path}...")
    val_df[["series_id", "timestamp"]].to_csv(local_val_index_path, index=False)
    
    print("Data split completed successfully!")

if __name__ == "__main__":
    main()
