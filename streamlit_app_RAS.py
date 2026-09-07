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
