"""
RAS Stabilization Prediction Model
====================================
Predicts biofilter/nitrogen-cycle stabilization time in a Recirculating
Aquaculture System (RAS) using:
  1. A Monod-kinetics ODE system (ammonia -> nitrite -> nitrate via AOB/NOB)
  2. Monte Carlo simulation over uncertain initial conditions / parameters
  3. Local stability analysis via the Jacobian's eigenvalues
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field, asdict

# ----------------------------------------------------------------------
# 1. BASE CONDITIONS (this is  single configuration point)
# ----------------------------------------------------------------------
# State vector order: [NH3, NO2, NO3, X_AOB, X_NOB]
#   NH3, NO2, NO3   : mg/L (nitrogen species)
#   X_AOB           : ammonia-oxidizing bacteria biomass (mg/L, proxy units)
#   X_NOB           : nitrite-oxidizing bacteria biomass (mg/L, proxy units)

@dataclass
class BaseConditions:
    # --- initial state ---
    NH3_0: float = 5.0        # mg/L, starting ammonia spike
    NO2_0: float = 0.0        # mg/L
    NO3_0: float = 0.0        # mg/L
    X_AOB_0: float = 0.05     # mg/L, seed AOB biomass
    X_NOB_0: float = 0.05     # mg/L, seed NOB biomass

    # --- Monod kinetics parameters (defaults from nitrification literature) ---
    mu_max_AOB: float = 0.9   # 1/day, max growth rate of AOB
    mu_max_NOB: float = 1.0   # 1/day, max growth rate of NOB
    Ks_NH3: float = 0.5       # mg/L, half-saturation const for AOB on NH3
    Ks_NO2: float = 0.3       # mg/L, half-saturation const for NOB on NO2
    Y_AOB: float = 0.15       # yield coefficient (biomass per mg N oxidized)
    Y_NOB: float = 0.05       # yield coefficient
    b_AOB: float = 0.05       # 1/day, AOB decay rate
    b_NOB: float = 0.05       # 1/day, NOB decay rate

    # --- environmental modifiers ---
    temperature_C: float = 20.0
    DO_mgL: float = 6.0
    Ks_DO: float = 0.5        # half-saturation for DO limitation

    # --- system / operational ---
    flow_exchange_rate: float = 0.1   # 1/day, dilution/water-exchange rate
    NH3_load_rate: float = 0.0        # mg/L/day, continuous feed-derived input

    def initial_state(self):
        return np.array([self.NH3_0, self.NO2_0, self.NO3_0,
                          self.X_AOB_0, self.X_NOB_0])


BASE = BaseConditions()

# Distributions used for Monte Carlo sampling (mean, std) — edit freely.
# Keys must match BaseConditions field names.
PARAM_DISTRIBUTIONS = {
    "NH3_0":       (BASE.NH3_0, 1.0),
    "temperature_C": (BASE.temperature_C, 2.0),
    "mu_max_AOB":  (BASE.mu_max_AOB, 0.1),
    "mu_max_NOB":  (BASE.mu_max_NOB, 0.1),
    "DO_mgL":      (BASE.DO_mgL, 0.5),
}

# ----------------------------------------------------------------------
# 2. ODE SYSTEM (Monod kinetics)
# ----------------------------------------------------------------------

def temperature_factor(T, T_ref=20.0, theta=1.072):
    """Arrhenius-type temperature correction (common in nitrification models)."""
    return theta ** (T - T_ref)

def ras_dynamics(t, y, p: BaseConditions):
    NH3, NO2, NO3, X_AOB, X_NOB = np.maximum(y, 0)  # keep non-negative

    T_corr = temperature_factor(p.temperature_C)
    DO_lim = p.DO_mgL / (p.Ks_DO + p.DO_mgL)

    # Monod-limited specific growth rates
    mu_AOB = p.mu_max_AOB * (NH3 / (p.Ks_NH3 + NH3)) * DO_lim * T_corr
    mu_NOB = p.mu_max_NOB * (NO2 / (p.Ks_NO2 + NO2)) * DO_lim * T_corr

    # Substrate consumption tied to growth via yield coefficients
    dNH3 = -(mu_AOB / p.Y_AOB) * X_AOB - p.flow_exchange_rate * NH3 + p.NH3_load_rate
    dNO2 = (mu_AOB / p.Y_AOB) * X_AOB - (mu_NOB / p.Y_NOB) * X_NOB - p.flow_exchange_rate * NO2
    dNO3 = (mu_NOB / p.Y_NOB) * X_NOB - p.flow_exchange_rate * NO3

    dX_AOB = (mu_AOB - p.b_AOB) * X_AOB
    dX_NOB = (mu_NOB - p.b_NOB) * X_NOB

    return [dNH3, dNO2, dNO3, dX_AOB, dX_NOB]


def run_simulation(p: BaseConditions, t_span=(0, 60), n_points=600):
    """Solve the ODE system from t_span[0] to t_span[1] days."""
    t_eval = np.linspace(*t_span, n_points)
    sol = solve_ivp(ras_dynamics, t_span, p.initial_state(), args=(p,),
                     t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9)
    return sol

# ----------------------------------------------------------------------
# 3. STABILITY ANALYSIS — Jacobian & eigenvalues
# ----------------------------------------------------------------------

def numerical_jacobian(state, p: BaseConditions, eps=1e-6):
    """Finite-difference Jacobian of ras_dynamics at a given state."""
    n = len(state)
    J = np.zeros((n, n))
    f0 = np.array(ras_dynamics(0, state, p))
    for i in range(n):
        perturbed = state.copy()
        perturbed[i] += eps
        f1 = np.array(ras_dynamics(0, perturbed, p))
        J[:, i] = (f1 - f0) / eps
    return J

def is_stable(state, p: BaseConditions):
    """Returns (stable: bool, eigenvalues: ndarray)."""
    J = numerical_jacobian(state, p)
    eigvals = np.linalg.eigvals(J)
    stable = np.all(eigvals.real < 0)
    return stable, eigvals

# ----------------------------------------------------------------------
# 4. STABILIZATION TIME DEFINITION
# ----------------------------------------------------------------------

def find_stabilization_time(sol, nh3_threshold=0.5, no2_threshold=0.5):
    """
    First time index where NH3 and NO2 both drop below threshold
    AND stay below it for the rest of the simulation.
    """
    NH3, NO2 = sol.y[0], sol.y[1]
    below = (NH3 < nh3_threshold) & (NO2 < no2_threshold)
    # find first index after which 'below' is True for all remaining points
    for i in range(len(below)):
        if np.all(below[i:]):
            return sol.t[i]
    return np.nan  # never stabilized within simulated window

# ----------------------------------------------------------------------
# 5. MONTE CARLO DRIVER
# ----------------------------------------------------------------------

def sample_conditions(base: BaseConditions, distributions: dict, rng):
    """Draw one randomized BaseConditions instance from distributions."""
    kwargs = asdict(base)
    for key, (mean, std) in distributions.items():
        kwargs[key] = max(0.0, rng.normal(mean, std))  # clip negatives
    return BaseConditions(**kwargs)

def monte_carlo_run(n_runs=500, t_span=(0, 60), seed=42):
    rng = np.random.default_rng(seed)
    stabilization_times = []
    stability_flags = []

    for _ in range(n_runs):
        p = sample_conditions(BASE, PARAM_DISTRIBUTIONS, rng)
        sol = run_simulation(p, t_span=t_span)
        t_stab = find_stabilization_time(sol)
        stabilization_times.append(t_stab)

        # Check stability at the final simulated state
        final_state = sol.y[:, -1]
        stable, _ = is_stable(final_state, p)
        stability_flags.append(stable)

    return np.array(stabilization_times), np.array(stability_flags)

# ----------------------------------------------------------------------
# 6. EXAMPLE USAGE (run this cell in Jupyter)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- single deterministic run with BASE conditions ---
    sol = run_simulation(BASE)
    t_stab = find_stabilization_time(sol)
    final_state = sol.y[:, -1]
    stable, eigvals = is_stable(final_state, BASE)

    print(f"Deterministic stabilization time: {t_stab:.1f} days")
    print(f"Stable at final state: {stable}")
    print(f"Eigenvalues: {eigvals}")

    # --- Monte Carlo over uncertainty ---
    times, flags = monte_carlo_run(n_runs=200)
    valid = times[~np.isnan(times)]
    print(f"\nMonte Carlo (n=200):")
    print(f"  Stabilized in window: {len(valid)}/200")
    if len(valid) > 0:
        print(f"  Mean stabilization time: {valid.mean():.1f} days")
        print(f"  90% range: {np.percentile(valid, 5):.1f} - {np.percentile(valid, 95):.1f} days")
    print(f"  Fraction stable at final state: {flags.mean():.2%}")
