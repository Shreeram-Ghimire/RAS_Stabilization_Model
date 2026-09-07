"""
Streamlit front-end for the RAS Stabilization Prediction Model.

Run locally with:
    pip install streamlit matplotlib
    streamlit run streamlit_app.py

Requires ras_stabilization_model.py in the same folder.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
from dataclasses import dataclass
from typing import Tuple

from ras_stabilization_model import (
    BaseConditions, run_simulation, find_stabilization_time,
    is_stable, monte_carlo_run, sample_conditions
)

st.set_page_config(page_title="RAS Stabilization Model", layout="wide")
st.title("RAS Stabilization Prediction Model")
st.caption("Monod-kinetics biofilter cycling model with Monte Carlo uncertainty "
           "and Jacobian stability analysis.")

# ------------------------------------------------------------------
# SIDEBAR — base conditions (mirrors the BaseConditions dataclass)
# ------------------------------------------------------------------
st.sidebar.header("Initial State")
NH3_0 = st.sidebar.number_input("Initial NH3 (mg/L)", 0.0, 50.0, 5.0, 0.5)
NO2_0 = st.sidebar.number_input("Initial NO2 (mg/L)", 0.0, 50.0, 0.0, 0.5)
NO3_0 = st.sidebar.number_input("Initial NO3 (mg/L)", 0.0, 50.0, 0.0, 0.5)
X_AOB_0 = st.sidebar.number_input("Initial AOB biomass (mg/L)", 0.0, 5.0, 0.05, 0.01)
X_NOB_0 = st.sidebar.number_input("Initial NOB biomass (mg/L)", 0.0, 5.0, 0.05, 0.01)

st.sidebar.header("Kinetic Parameters")
mu_max_AOB = st.sidebar.slider("mu_max AOB (1/day)", 0.1, 2.0, 0.9, 0.05)
mu_max_NOB = st.sidebar.slider("mu_max NOB (1/day)", 0.1, 2.0, 1.0, 0.05)
Ks_NH3 = st.sidebar.slider("Ks NH3 (mg/L)", 0.05, 3.0, 0.5, 0.05)
Ks_NO2 = st.sidebar.slider("Ks NO2 (mg/L)", 0.05, 3.0, 0.3, 0.05)
Y_AOB = st.sidebar.slider("Yield AOB", 0.01, 0.5, 0.15, 0.01)
Y_NOB = st.sidebar.slider("Yield NOB", 0.01, 0.5, 0.05, 0.01)
b_AOB = st.sidebar.slider("Decay rate AOB (1/day)", 0.0, 0.3, 0.05, 0.01)
b_NOB = st.sidebar.slider("Decay rate NOB (1/day)", 0.0, 0.3, 0.05, 0.01)

st.sidebar.header("Environment / Operations")
temperature_C = st.sidebar.slider("Temperature (°C)", 5.0, 35.0, 20.0, 0.5)
DO_mgL = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.5, 12.0, 6.0, 0.5)
flow_exchange_rate = st.sidebar.slider("Water exchange rate (1/day)", 0.0, 1.0, 0.1, 0.01)
NH3_load_rate = st.sidebar.slider("Continuous NH3 load (mg/L/day)", 0.0, 5.0, 0.0, 0.1)

st.sidebar.header("Simulation Window")
t_max = st.sidebar.slider("Days to simulate", 5, 180, 60, 5)

base = BaseConditions(
    NH3_0=NH3_0, NO2_0=NO2_0, NO3_0=NO3_0, X_AOB_0=X_AOB_0, X_NOB_0=X_NOB_0,
    mu_max_AOB=mu_max_AOB, mu_max_NOB=mu_max_NOB, Ks_NH3=Ks_NH3, Ks_NO2=Ks_NO2,
    Y_AOB=Y_AOB, Y_NOB=Y_NOB, b_AOB=b_AOB, b_NOB=b_NOB,
    temperature_C=temperature_C, DO_mgL=DO_mgL,
    flow_exchange_rate=flow_exchange_rate, NH3_load_rate=NH3_load_rate,
)

# ------------------------------------------------------------------
# Helper functions for downloads
# ------------------------------------------------------------------
def create_excel_download(sol, t_stab, stable, eigvals, base_params):
    """Create an Excel file with simulation data and metadata."""
    # Create DataFrame with time series data
    labels = ["NH3", "NO2", "NO3", "X_AOB", "X_NOB"]
    data_dict = {"Time (days)": sol.t}
    for i, lab in enumerate(labels):
        data_dict[lab] = sol.y[i]
    
    df = pd.DataFrame(data_dict)
    
    # Create metadata DataFrame
    metadata = {
        "Parameter": ["Stabilization Time (days)", "Stable at Final State", 
                      "Max Eigenvalue (Real Part)", "Simulation Date"],
        "Value": [t_stab if not np.isnan(t_stab) else "Not reached", 
                  "Yes" if stable else "No",
                  f"{eigvals.real.max():.6f}",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }
    metadata_df = pd.DataFrame(metadata)
    
    # Create base conditions DataFrame
    base_df = pd.DataFrame({
        "Parameter": list(base_params.__dict__.keys()),
        "Value": list(base_params.__dict__.values())
    })
    
    # Write to Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Simulation Data', index=False)
        metadata_df.to_excel(writer, sheet_name='Summary', index=False)
        base_df.to_excel(writer, sheet_name='Parameters', index=False)
    
    return output.getvalue()

def create_dashboard_download(sol, t_stab, stable, eigvals, base_params, fig):
    """Create an HTML dashboard with the simulation results."""
    # This creates a self-contained HTML file with the plot and results
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>RAS Stabilization Model - Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric {{ background: #ecf0f1; padding: 15px; border-radius: 8px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            .plot-container {{ margin: 30px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐟 RAS Stabilization Model - Dashboard</h1>
            <p><em>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Stabilization Time</div>
                    <div class="metric-value">{t_stab:.1f} days</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Stable at Final State</div>
                    <div class="metric-value">{'✓ Yes' if stable else '✗ No'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Eigenvalue (Real Part)</div>
                    <div class="metric-value">{eigvals.real.max():.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Final State - NH3</div>
                    <div class="metric-value">{sol.y[0, -1]:.3f} mg/L</div>
                </div>
            </div>
            
            <h2>Simulation Results</h2>
            <div class="plot-container">
                <!-- Plot will be embedded as base64 image -->
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Simulation Plot">
            </div>
            
            <h2>Final State Values</h2>
            <table>
                <thead>
                    <tr><th>Component</th><th>Final Concentration (mg/L)</th></tr>
                </thead>
                <tbody>
                    <tr><td>NH3</td><td>{sol.y[0, -1]:.6f}</td></tr>
                    <tr><td>NO2</td><td>{sol.y[1, -1]:.6f}</td></tr>
                    <tr><td>NO3</td><td>{sol.y[2, -1]:.6f}</td></tr>
                    <tr><td>X_AOB</td><td>{sol.y[3, -1]:.6f}</td></tr>
                    <tr><td>X_NOB</td><td>{sol.y[4, -1]:.6f}</td></tr>
                </tbody>
            </table>
            
            <h2>Model Parameters</h2>
            <table>
                <thead>
                    <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
    """
    
    # Add parameters to table
    for key, value in base_params.__dict__.items():
        html_content += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
    
    html_content += f"""
                </tbody>
            </table>
            
            <div class="footer">
                Generated by RAS Stabilization Prediction Model • {datetime.now().strftime("%Y-%m-%d")}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_download_link(data, filename, mime_type):
    """Generate a download link for the given data."""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}" style="text-decoration: none; padding: 10px 20px; background-color: #4CAF50; color: white; border-radius: 5px; border: none; cursor: pointer; display: inline-block;">📥 Download {filename}</a>'
    
# ------------------------------------------------------------------
# MAIN — deterministic run
# ------------------------------------------------------------------
tab1, tab2 = st.tabs(["Single Run", "Monte Carlo"])

with tab1:
    sol = run_simulation(base, t_span=(0, t_max))
    t_stab = find_stabilization_time(sol)
    final_state = sol.y[:, -1]
    stable, eigvals = is_stable(final_state, base)

    col1, col2, col3 = st.columns(3)
    col1.metric("Stabilization time",
                f"{t_stab:.1f} days" if not np.isnan(t_stab) else "Not reached")
    col2.metric("Stable at final state?", "Yes" if stable else "No")
    col3.metric("Max eigenvalue (real part)", f"{eigvals.real.max():.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = ["NH3", "NO2", "NO3", "X_AOB", "X_NOB"]
    for i, lab in enumerate(labels):
        ax.plot(sol.t, sol.y[i], label=lab)
    if not np.isnan(t_stab):
        ax.axvline(t_stab, color="gray", linestyle="--", alpha=0.6, label="Stabilized")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Concentration / biomass (mg/L)")
    ax.legend()
    ax.set_title("RAS nitrogen cycle dynamics")
    st.pyplot(fig)

    with st.expander("Eigenvalues (Jacobian at final state)"):
        st.write(eigvals)

with tab2:
    st.write("Runs the model repeatedly, randomizing NH3_0, temperature, "
             "mu_max_AOB/NOB, and DO around the sidebar values (std devs are "
             "fixed in `ras_stabilization_model.PARAM_DISTRIBUTIONS` — edit "
             "that dict to change spread).")
    n_runs = st.slider("Number of Monte Carlo runs", 20, 1000, 200, 20)
    run_button = st.button("Run Monte Carlo simulation")

    if run_button:
        with st.spinner(f"Running {n_runs} simulations..."):
            # temporarily point the module's BASE at the sidebar-configured values
            import ras_stabilization_model as ras_mod
            ras_mod.BASE = base
            times, flags = monte_carlo_run(n_runs=n_runs, t_span=(0, t_max))

        valid = times[~np.isnan(times)]
        c1, c2, c3 = st.columns(3)
        c1.metric("Stabilized within window", f"{len(valid)}/{n_runs}")
        c2.metric("Mean stabilization time",
                  f"{valid.mean():.1f} days" if len(valid) else "—")
        c3.metric("Fraction stable at final state", f"{flags.mean():.1%}")

        if len(valid) > 0:
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            ax2.hist(valid, bins=30, edgecolor="black", alpha=0.75)
            ax2.set_xlabel("Stabilization time (days)")
            ax2.set_ylabel("Number of simulations")
            ax2.set_title("Distribution of predicted stabilization time")
            st.pyplot(fig2)

            p5, p50, p95 = np.percentile(valid, [5, 50, 95])
            st.write(f"**Median:** {p50:.1f} days  |  **90% range:** {p5:.1f}–{p95:.1f} days")
        else:
            st.warning("No runs stabilized within the simulated window — try a longer window "
                       "or check parameter ranges.")
