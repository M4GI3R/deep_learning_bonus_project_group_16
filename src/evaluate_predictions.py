import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    smape = float(200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true) + 1e-8))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "sMAPE (%)": smape,
        "WAPE": wape
    }

def main():
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / "output"
    dataset_dir = project_root / "res" / "dataset"
    gt_path = dataset_dir / "local_validation_targets.csv"
    
    if not gt_path.exists():
        print(f"Error: Local validation targets ground truth file not found at {gt_path}")
        print("Please split the dataset first: uv run python src/split_data.py")
        return
        
    print(f"Loading local ground truth targets from {gt_path}...")
    gt_df = pd.read_csv(gt_path)
    gt_df["timestamp"] = pd.to_datetime(gt_df["timestamp"])
    
    # Discover prediction CSV files in output/ recursively
    print(f"Scanning {output_dir} for prediction files...")
    prediction_files = []
    
    for file in output_dir.rglob("*.csv"):
        # Ignore files that are obviously not predictions
        if file.name.startswith("metrics"):
            continue
            
        try:
            # Quick validation of schema
            df_head = pd.read_csv(file, nrows=2)
            required_cols = {"series_id", "timestamp"}
            if not required_cols.issubset(df_head.columns):
                continue
            # Must have either "prediction" or "target" (ground truth)
            if "prediction" not in df_head.columns and "target" not in df_head.columns:
                continue
                
            # Determine directory structure
            rel_path = file.relative_to(output_dir)
            
            if file.name.lower() in ["predictions.csv", "prediction.csv"]:
                # Nested layout: output/<category>/<model_name>/predictions.csv
                model_name = file.parent.name
                category = file.parent.parent.name if file.parent.parent != output_dir else "Root"
                metrics_path = file.parent / "metrics.json"
            else:
                # Flat layout: output/<category>/<model_name>.csv
                model_name = file.stem
                category = file.parent.name if file.parent != output_dir else "Root"
                metrics_path = file.parent / f"{file.stem}_metrics.json"
                
            display_name = f"{category} / {model_name}" if category != "Root" else model_name
            
            prediction_files.append({
                "name": model_name,
                "category": category,
                "display_name": display_name,
                "path": file,
                "metrics_path": metrics_path
            })
        except Exception:
            # Not a valid CSV or permission error
            continue
        
    if not prediction_files:
        print("No prediction files found in the output directory.")
        return
        
    print(f"Found {len(prediction_files)} prediction files. Evaluating...")
    
    results = {}
    
    # 1. First Pass: Compute basic metrics for all discovered predictions
    for pred in prediction_files:
        print(f"Evaluating {pred['display_name']}...")
        try:
            pred_df = pd.read_csv(pred["path"])
            pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
            
            # Align predictions with ground truth
            # Supporting prediction column named "prediction" or "target"
            pred_col = "prediction" if "prediction" in pred_df.columns else "target"
            
            aligned = pd.merge(gt_df, pred_df, on=["series_id", "timestamp"], how="inner")
            
            if aligned.empty:
                print(f"Warning: No aligned timestamps found for {pred['display_name']}. Skipping.")
                continue
                
            aligned = aligned.sort_values(by=["series_id", "timestamp"]).reset_index(drop=True)
            
            # Global Metrics
            global_scores = calculate_metrics(aligned["target"].to_numpy(), aligned[pred_col].to_numpy())
            
            # Per-Series sMAPE (for Boxplot)
            series_smapes = {}
            for series_id, s_group in aligned.groupby("series_id"):
                y_t = s_group["target"].to_numpy()
                y_p = s_group[pred_col].to_numpy()
                series_smapes[series_id] = float(200 * np.mean(np.abs(y_t - y_p) / (np.abs(y_t) + np.abs(y_p) + 1e-8)))
                
            # Horizon Error (MAE per step)
            aligned["step"] = aligned.groupby("series_id").cumcount() + 1
            horizon_mae = {}
            for step, step_group in aligned.groupby("step"):
                y_t = step_group["target"].to_numpy()
                y_p = step_group[pred_col].to_numpy()
                horizon_mae[int(step)] = float(np.mean(np.abs(y_t - y_p)))
                
            results[pred["display_name"]] = {
                "name": pred["name"],
                "category": pred["category"],
                "display_name": pred["display_name"],
                "global_metrics": global_scores,
                "per_series_smape": series_smapes,
                "horizon_mae": horizon_mae,
                "metrics_path": pred["metrics_path"]
            }
        except Exception as e:
            print(f"Error evaluating {pred['display_name']}: {e}")
            
    # 2. Second Pass: Calculate WAPE Improvement and save individual JSON files
    for d_name, res in results.items():
        cat = res["category"]
        naive_key = f"{cat} / naive_last_value" if cat != "Root" else "naive_last_value"
        
        improvement = None
        if naive_key in results:
            naive_wape = results[naive_key]["global_metrics"]["WAPE"]
            model_wape = res["global_metrics"]["WAPE"]
            if naive_wape > 0:
                improvement = float((naive_wape - model_wape) / naive_wape * 100)
                
        res["global_metrics"]["WAPE Improvement (%)"] = improvement
        
        # Save metrics to the individual path, excluding the path key itself
        model_metrics = {
            "name": res["name"],
            "category": res["category"],
            "display_name": res["display_name"],
            "global_metrics": res["global_metrics"],
            "per_series_smape": res["per_series_smape"],
            "horizon_mae": res["horizon_mae"]
        }
        
        metrics_file = res["metrics_path"]
        print(f"Saving metrics for {d_name} to {metrics_file}...")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(model_metrics, f, indent=2)
            
    print("Pre-computation of model-specific metrics complete!")

if __name__ == "__main__":
    main()
