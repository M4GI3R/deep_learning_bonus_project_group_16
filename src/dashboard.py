import streamlit as st
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page config for professional matrix layout
st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Screenshot-Ready Matrix tables
st.markdown("""
<style>
    /* Typography and Layout */
    .header-style { font-size: 26px; font-weight: 600; color: #FFFFFF; margin-bottom: 2px; }
    .subheader-style { font-size: 13px; font-weight: 500; color: #00ADB5; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 20px;}
    .section-title { font-size: 18px; font-weight: 600; color: #EEEEEE; margin-top: 30px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    
    /* Table Styling for Screenshot Readiness */
    .se-table-container { width: 100%; overflow-x: auto; margin-bottom: 25px; border-radius: 6px; border: 1px solid #333; background-color: #121418; }
    .se-table { width: 100%; border-collapse: collapse; font-family: 'Inter', 'Segoe UI', monospace; font-size: 13.5px; }
    .se-table th { background-color: #1A1D24; color: #00ADB5; text-align: left; padding: 12px 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid #222831; }
    .se-table td { padding: 10px 15px; border-bottom: 1px solid #222831; color: #D3D6DA; }
    .se-table tr:hover { background-color: #1A1D24; }
    
    /* Checkbox Group Styling in Sidebar */
    .sidebar-cgroup { font-size: 14px; font-weight: 600; color: #EEEEEE; margin-top: 20px; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-style'>Evaluation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Model Output Evaluation & Comparative Analysis</div>", unsafe_allow_html=True)

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
                # Verify metrics schema
                if "global_metrics" in data and "horizon_mae" in data:
                    display_name = data.get("display_name", file.stem)
                    results[display_name] = data
            except Exception:
                continue
    return results

metrics_data = discover_and_load_metrics(OUTPUTS_ROOT)

if not metrics_data:
    st.warning("⚠️ Operations Matrix Offline.")
    st.info("No precomputed metrics.json files located in output/ subdirectories.")
    st.info("Please run the metrics precomputation command in your WSL terminal first:")
    st.code("uv run python src/evaluate_predictions.py", language="bash")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar - Model Selection Registry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='header-style' style='font-size: 20px;'>Matrix Selection</div>", unsafe_allow_html=True)
    st.caption("Select models to include in the visual comparisons below.")
    
    # Group runs by category
    grouped_models = {}
    for display_name, m_info in metrics_data.items():
        cat = m_info["category"]
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
    st.info("Engage at least one operation stage from the sidebar matrix.")
    st.stop()

# ---------------------------------------------------------------------------
# 1. Screenshot-Ready Evaluation Matrix Table
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Operations Leaderboard Matrix</div>", unsafe_allow_html=True)

# The leaderboard always contains every evaluated model.  The sidebar only
# controls the visual comparisons, so reference baselines such as
# ``seasonal_mean`` cannot disappear from the table.
leaderboard_models = sorted(
    metrics_data,
    key=lambda display_name: metrics_data[display_name]["global_metrics"]["MAE"],
)

# ``naive_last_value`` is the sole reference baseline, irrespective of its
# output category.  This keeps all models comparable to the same benchmark.
baseline_name = next(
    (
        display_name
        for display_name, model_info in metrics_data.items()
        if model_info["name"] == "naive_last_value"
    ),
    None,
)
baseline_wape = (
    metrics_data[baseline_name]["global_metrics"]["WAPE"]
    if baseline_name is not None
    else None
)

# Construct table rows
html_table = "<div class='se-table-container'><table class='se-table'>"
html_table += "<thead><tr>"
html_table += "<th>Rank</th>"
html_table += "<th>Category</th>"
html_table += "<th>Model Name</th>"
html_table += "<th>MAE</th>"
html_table += "<th>RMSE</th>"
html_table += "<th>sMAPE (%)</th>"
html_table += "<th>WAPE</th>"
html_table += "<th>WAPE Improvement vs Naive (%)</th>"
html_table += "</tr></thead><tbody>"

# Highlight the minimum value in every error-metric column.  Ties are all
# highlighted, rather than selecting an arbitrary single winner.
metric_columns = ("MAE", "RMSE", "sMAPE (%)", "WAPE")
best_metric_values = {
    metric: min(metrics_data[d_name]["global_metrics"][metric] for d_name in leaderboard_models)
    for metric in metric_columns
}

for rank, d_name in enumerate(leaderboard_models, start=1):
    m_info = metrics_data[d_name]
    metrics = m_info["global_metrics"]
    
    cat = m_info["category"]
    name = m_info["name"]
    mae = metrics["MAE"]
    rmse = metrics["RMSE"]
    smape = metrics["sMAPE (%)"]
    wape = metrics["WAPE"]
    is_baseline = m_info["name"] == "naive_last_value"
    improvement = None
    if baseline_wape is not None and baseline_wape > 0:
        improvement = (baseline_wape - wape) / baseline_wape * 100

    # Only the naïve last-value reference receives whole-row highlighting.
    row_style = (
        "style='background-color: rgba(241, 196, 15, 0.14); font-weight: bold;'"
        if is_baseline
        else ""
    )
        
    html_table += f"<tr {row_style}>"
    html_table += f"<td>{rank}</td>"
    html_table += f"<td style='color:#00ADB5;'><i>{cat}</i></td>"
    html_table += f"<td><b>{name}</b></td>"
    mae_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(mae, best_metric_values["MAE"]) else ""
    rmse_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(rmse, best_metric_values["RMSE"]) else ""
    smape_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(smape, best_metric_values["sMAPE (%)"]) else ""
    wape_style = "style='color: #2ECC71; font-weight: 700;'" if np.isclose(wape, best_metric_values["WAPE"]) else ""
    html_table += f"<td {mae_style}>{mae:.4f}</td>"
    html_table += f"<td {rmse_style}>{rmse:.4f}</td>"
    html_table += f"<td {smape_style}>{smape:.2f}%</td>"
    html_table += f"<td {wape_style}>{wape:.4f}</td>"

    if is_baseline:
        imp_html = "<span style='color: #F1C40F;'>Baseline (Naive)</span>"
    elif improvement is None:
        imp_html = "<span style='color: #888888;'>Unavailable</span>"
    elif improvement > 0:
        imp_html = f"<span style='color: #2ECC71;'>+{improvement:.2f}%</span>"
    elif improvement < 0:
        imp_html = f"<span style='color: #E74C3C;'>{improvement:.2f}%</span>"
    else:
        imp_html = "<span style='color: #888888;'>0.00%</span>"
        
    html_table += f"<td>{imp_html}</td>"
    html_table += "</tr>"

html_table += "</tbody></table></div>"
st.markdown(html_table, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. Visual Analytics Section (Side-by-Side Plots)
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Visual Telemetry & Drift Analysis</div>", unsafe_allow_html=True)

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.markdown("### Horizon Error Drift (MAE vs Step)")
    st.caption("Lower is better. Demonstrates error accumulation over the 336-hour forecast rollout.")
    
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
            name=m_info["name"],
            line=dict(width=2)
        ))
        
    drift_fig.update_layout(
        template="plotly_dark",
        xaxis_title="Rollout Step (Hours Ahead)",
        yaxis_title="Mean Absolute Error (MAE)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    st.plotly_chart(drift_fig, width='stretch')

with plot_col2:
    st.markdown("### Error Dispersion Across Series (sMAPE Distribution)")
    st.caption("Displays the spread of forecasting accuracy across the 96 operational units.")
    
    box_fig = go.Figure()
    
    for d_name in selected_models:
        m_info = metrics_data[d_name]
        series_smapes = m_info["per_series_smape"]
        
        smapes_values = list(series_smapes.values())
        
        box_fig.add_trace(go.Box(
            y=smapes_values,
            name=m_info["name"],
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
