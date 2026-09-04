import streamlit as st
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styling
st.markdown("""
<style>
    /* Typography and Layout */
    .header-style { font-size: 26px; font-weight: 600; color: #FFFFFF; margin-bottom: 2px; }
    .subheader-style { font-size: 13px; font-weight: 500; color: #00ADB5; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px;}
    .section-title { font-size: 18px; font-weight: 600; color: #EEEEEE; margin-top: 30px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    
    /* Results table */
    .se-table-container { width: 100%; overflow-x: auto; margin-bottom: 25px; border-radius: 6px; border: 1px solid #333; background-color: #121418; }
    .se-table { width: 100%; border-collapse: collapse; font-family: 'Inter', 'Segoe UI', monospace; font-size: 13.5px; }
    .se-table th { background-color: #1A1D24; color: #00ADB5; text-align: left; padding: 12px 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid #222831; }
    .se-table td { padding: 10px 15px; border-bottom: 1px solid #222831; color: #D3D6DA; }
    .se-table tbody tr { background-color: #15181E; transition: background-color 140ms ease; }
    .se-table tbody tr:nth-child(even) { background-color: #191D24; }
    .se-table tbody tr.row-tcn { background-color: #10292C; }
    .se-table tbody tr.row-dlinear { background-color: #281F32; }
    .se-table tbody tr.row-baseline { background-color: #2B2817; }
    .se-table tbody tr:hover { background-color: #1B3034; }
    
    /* Checkbox Group Styling in Sidebar */
    .sidebar-cgroup { font-size: 14px; font-weight: 600; color: #EEEEEE; margin-top: 20px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-style'>Evaluation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Local Validation and Comparative Error Analysis</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths & Loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = BASE_DIR / "output"

@st.cache_data(ttl=10)
def discover_and_load_metrics(output_path: Path):
    """Scan outputs directory for metrics.json or *_metrics.json and load them."""
    results = {}
    if not output_path.exists():
        return results
        
    for file in output_path.rglob("*.json"):
        # Ensure it is a valid model-specific metrics file
        if file.name == "metrics.json" or file.name.endswith("_metrics.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                required_fields = {
                    "name",
                    "category",
                    "global_metrics",
                    "per_series_smape",
                    "horizon_mae",
                }
                missing_fields = required_fields.difference(data)
                if missing_fields:
                    raise ValueError(f"missing fields: {sorted(missing_fields)}")
                required_metrics = {
                    "MAE", "MSE", "RMSE", "MAPE (%)", "sMAPE (%)", "WAPE"
                }
                missing_metrics = required_metrics.difference(data["global_metrics"])
                if missing_metrics:
                    raise ValueError(f"missing metrics: {sorted(missing_metrics)}")
                if data.get("metric_scale") != "leaderboard_percent_v1":
                    # Earlier evaluator versions stored WAPE as a fraction;
                    # current metric files store it as a percentage.
                    data["global_metrics"]["WAPE"] *= 100.0
                relative_parts = file.relative_to(output_path).parts
                known_datasets = {
                    "operations_forecasting_2026",
                    "electricity_load_diagrams",
                }
                data["_dataset"] = (
                    relative_parts[0]
                    if relative_parts and relative_parts[0] in known_datasets
                    else "legacy"
                )
                # The same model name occurs across datasets and feature
                # sets. Use its output path as the internal key so one metrics
                # file can never overwrite another.
                run_key = file.relative_to(output_path).as_posix()
                results[run_key] = data
            except Exception as exc:
                print(f"Skipping invalid metrics file {file}: {exc}")
                continue
    return results

metrics_data = discover_and_load_metrics(OUTPUTS_ROOT)

if not metrics_data:
    st.warning("No evaluation metrics found.")
    st.info("No precomputed metrics.json files located in output/ subdirectories.")
    st.info("Please run the metrics precomputation command in your WSL terminal first:")
    st.code("uv run python src/evaluate_predictions.py", language="bash")
    st.stop()


def compact_category(model_info: dict) -> str:
    """Remove dataset and horizon boilerplate from dashboard categories."""
    parts = [part.strip() for part in model_info["category"].split(" / ")]
    if "baselines" in parts:
        return "baseline"
    if model_info["_dataset"] == "operations_forecasting_2026":
        return "model"
    experiment_labels = {
        "operations_calendar": "matched",
        "calendar_extended": "extended",
    }
    return experiment_labels.get(parts[-1], parts[-1])


def format_native_error(value: float) -> str:
    """Format absolute errors without changing their dataset-native scale."""
    value = float(value)
    decimals = 4 if abs(value) < 100 else 2
    return f"{value:,.{decimals}f}"

# ---------------------------------------------------------------------------
# Sidebar - Model Selection Registry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='header-style' style='font-size: 20px;'>Display Options</div>", unsafe_allow_html=True)
    st.caption("Select a dataset and the runs to include in the plots.")

    dataset_labels = {
        "operations_forecasting_2026": "Operations Forecasting",
        "electricity_load_diagrams": "Electricity Load Diagrams",
        "legacy": "Legacy / Ungrouped",
    }
    available_datasets = sorted(
        {info["_dataset"] for info in metrics_data.values()},
        key=lambda value: dataset_labels.get(value, value),
    )
    selected_dataset = st.selectbox(
        "Dataset",
        available_datasets,
        format_func=lambda value: dataset_labels.get(value, value),
    )
    metrics_data = {
        name: info
        for name, info in metrics_data.items()
        if info["_dataset"] == selected_dataset
    }

    # Group runs by category
    grouped_models = {}
    for display_name, m_info in metrics_data.items():
        cat = compact_category(m_info)
        if cat not in grouped_models:
            grouped_models[cat] = []
        grouped_models[cat].append((display_name, m_info["name"]))
        
    selected_models = []
    
    for cat_name, models in grouped_models.items():
        st.markdown(f"<div class='sidebar-cgroup'>{cat_name}</div>", unsafe_allow_html=True)
        for display_name, m_name in models:
            # Default to checked
            is_checked = st.checkbox(m_name, value=True, key=display_name)
            if is_checked:
                selected_models.append(display_name)

if not selected_models:
    st.info("Select at least one model in the sidebar.")
    st.stop()

if selected_dataset == "electricity_load_diagrams":
    stale_metrics = [
        info["name"]
        for info in metrics_data.values()
        if info.get("mape_zero_policy") != "exclude_zero_actuals"
    ]
    if stale_metrics:
        st.error(
            "Electricity metrics were generated with the obsolete zero-dividing "
            "MAPE formula. Recompute metrics from the saved predictions; model "
            "retraining is not required."
        )
        st.code(
            "uv run python src/evaluate_predictions.py --dataset electricity",
            language="bash",
        )
        st.stop()

# ---------------------------------------------------------------------------
# 1. Evaluation table
# ---------------------------------------------------------------------------
selected_dataset_label = dataset_labels.get(selected_dataset, selected_dataset)
st.markdown(
    f"<div class='section-title'>{selected_dataset_label} Local Validation Matrix</div>",
    unsafe_allow_html=True,
)
st.caption(
    f"Ranks and best-value highlights use only {selected_dataset_label} runs. "
    "MAE and RMSE remain in this dataset's native target units; MSE remains in "
    "squared target units. No cross-dataset scaling is applied."
)

# Produce a within-dataset aggregate rank by percentile-ranking every available
# error metric and averaging the six percentile ranks.
metric_columns = ("MAE", "MSE", "RMSE", "MAPE (%)", "sMAPE (%)", "WAPE")
rank_frame = pd.DataFrame(
    {
        display_name: {
            metric: metrics_data[display_name]["global_metrics"][metric]
            for metric in metric_columns
        }
        for display_name in metrics_data
    }
).T
overall_rank_scores = pd.concat(
    [
        rank_frame[metric].rank(method="average", pct=True, ascending=True)
        for metric in metric_columns
    ],
    axis=1,
).mean(axis=1)

# The leaderboard always contains every evaluated model. The sidebar only
# controls plots, so reference baselines cannot disappear from the table.
leaderboard_models = sorted(
    metrics_data,
    key=lambda display_name: overall_rank_scores.loc[display_name],
)

# Construct table rows
html_table = "<div class='se-table-container'><table class='se-table'>"
html_table += "<thead><tr>"
html_table += "<th>Rank</th>"
html_table += "<th>Category</th>"
html_table += "<th>Model Name</th>"
html_table += "<th>MAE</th>"
html_table += "<th>MSE</th>"
html_table += "<th>RMSE</th>"
html_table += "<th>MAPE (%)</th>"
html_table += "<th>sMAPE (%)</th>"
html_table += "<th>WAPE (%)</th>"
html_table += "</tr></thead><tbody>"

# Highlight the minimum value in every error-metric column. Ties are all
# highlighted rather than selecting an arbitrary winner.
best_metric_values = {
    metric: min(metrics_data[d_name]["global_metrics"][metric] for d_name in leaderboard_models)
    for metric in metric_columns
}

for rank, d_name in enumerate(leaderboard_models, start=1):
    m_info = metrics_data[d_name]
    metrics = m_info["global_metrics"]
    
    cat = compact_category(m_info)
    name = m_info["name"]
    mae = metrics["MAE"]
    mse = metrics["MSE"]
    rmse = metrics["RMSE"]
    mape = metrics["MAPE (%)"]
    smape = metrics["sMAPE (%)"]
    wape = metrics["WAPE"]
    if name.lower() == "tcn":
        row_class = "row-tcn"
    elif name.lower() == "dlinear":
        row_class = "row-dlinear"
    else:
        row_class = "row-baseline"
        
    html_table += f"<tr class='{row_class}'>"
    html_table += f"<td>{rank}</td>"
    html_table += f"<td style='color:#00ADB5;'><i>{cat}</i></td>"
    html_table += f"<td><b>{name}</b></td>"
    mae_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(mae, best_metric_values["MAE"]) else ""
    mse_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(mse, best_metric_values["MSE"]) else ""
    rmse_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(rmse, best_metric_values["RMSE"]) else ""
    mape_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(mape, best_metric_values["MAPE (%)"]) else ""
    smape_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(smape, best_metric_values["sMAPE (%)"]) else ""
    wape_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(wape, best_metric_values["WAPE"]) else ""
    html_table += f"<td {mae_style}>{format_native_error(mae)}</td>"
    html_table += f"<td {mse_style}>{format_native_error(mse)}</td>"
    html_table += f"<td {rmse_style}>{format_native_error(rmse)}</td>"
    html_table += f"<td {mape_style}>{mape:.2f}%</td>"
    html_table += f"<td {smape_style}>{smape:.2f}%</td>"
    html_table += f"<td {wape_style}>{wape:.2f}%</td>"
    html_table += "</tr>"

html_table += "</tbody></table></div>"
st.markdown(html_table, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Diagnostic plots
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Error Analysis</div>", unsafe_allow_html=True)

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.markdown("### Forecast Horizon Error (MAE vs Step)")
    st.caption("Lower is better. Direct models predict all 336 hours without target rollout.")
    
    drift_fig = go.Figure()
    
    for d_name in selected_models:
        m_info = metrics_data[d_name]
        horizon_mae = m_info["horizon_mae"]
        
        # Sort steps numerically
        sorted_steps = sorted([int(k) for k in horizon_mae.keys()])
        mae_values = [horizon_mae[str(step)] for step in sorted_steps]
        
        drift_fig.add_trace(go.Scatter(
            x=sorted_steps,
            y=mae_values,
            mode="lines",
            name=f"{compact_category(m_info)} / {m_info['name']}",
            line=dict(width=2)
        ))
        
    drift_fig.update_layout(
        template="plotly_dark",
        xaxis_title="Forecast Step (Hours Ahead)",
        yaxis_title="Mean Absolute Error (MAE)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    st.plotly_chart(drift_fig, width='stretch')

with plot_col2:
    st.markdown("### Error Dispersion Across Series (sMAPE Distribution)")
    series_count = max(
        len(metrics_data[name]["per_series_smape"]) for name in selected_models
    )
    st.caption(f"Displays the spread of forecasting accuracy across {series_count} series.")
    
    box_fig = go.Figure()
    
    for d_name in selected_models:
        m_info = metrics_data[d_name]
        series_smapes = m_info["per_series_smape"]
        
        smapes_values = list(series_smapes.values())
        
        box_fig.add_trace(go.Box(
            y=smapes_values,
            name=f"{compact_category(m_info)} / {m_info['name']}",
            boxpoints='outliers',
            jitter=0.3
        ))
        
    box_fig.update_layout(
        template="plotly_dark",
        yaxis_title="sMAPE (%) per Series",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(box_fig, width='stretch')
