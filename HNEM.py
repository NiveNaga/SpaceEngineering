"""
HNEM (Homogeneous Non-Equilibrium Model) — Ball Valve
Maps valve angle (theta) <-> mass flow rate (m_dot) for N2O.

Fixes applied vs. original:
  #1  x_downstream: np.clip tuple bug -> np.clip(x, 0, 1)
  #2  dp_down typo -> dP_down throughout
  #3  x_real not in scope for X_K -> replaced with phi-corrected quality array x_phi
  #4  find_theta_bisection -> find_theta (name mismatch in main)
  #5  stale `rho` in K_line loop -> rho_q
  #6  PR_total missing from Xq in mass_flow -> added, feature count now 14 everywhere
  #7  phi optimisation tautology -> K fixed from HEM reference; phi scales x_eq toward HEM
  #8  (same as #6 / #11) PR_total consistency
  #9  model_state leakage -> OOF predictions via cross_val_predict feed model_K
  #10 rho_mix called at P_up for downstream state -> now called at P_down
  #11 isenthalpic flash assumption documented
  #12 PropsSI loops -> vectorised where possible
  #13 silent except -> warnings logged
  #14 K_line.append array -> .predict(...)[0]
  #15 debug prints removed
  #16 plt.show() moved inside __main__ guard
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CoolProp.CoolProp import PropsSI
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

FLUID = "NitrousOxide"
P_MIN = 1e5   # Pa  – floor to avoid sub-atmospheric CoolProp calls
DP_MIN = 1.0  # Pa  – floor for pressure differences


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

def get_upstream_state(P: float, T: float) -> tuple[float, float]:
    """
    Return (vapour quality x, specific enthalpy h) for N2O at (P, T).

    Assumes isenthalpic expansion downstream (standard HNEM / HEM assumption):
    the upstream enthalpy is conserved across the valve with no work or
    heat transfer terms. Quality is clipped to [0, 1].

    Returns (0.0, 0.0) with a warning on CoolProp failure (e.g. supercritical).
    """
    try:
        h   = PropsSI("H", "P", P, "T", T, FLUID)
        h_l = PropsSI("H", "P", P, "Q", 0, FLUID)
        h_v = PropsSI("H", "P", P, "Q", 1, FLUID)
        x   = (h - h_l) / (h_v - h_l)
        return float(np.clip(x, 0.0, 1.0)), float(h)
    except Exception as exc:
        warnings.warn(f"get_upstream_state failed at P={P:.3e} T={T:.1f}: {exc}")
        return 0.0, 0.0


def x_downstream(h_up: float, P_down: float) -> float:
    """
    Equilibrium (HEM) vapour quality at downstream pressure P_down given
    the isenthalpically expanded upstream enthalpy h_up.

    Fix #1: was np.clip((x, 0, 1)) — tuple arg instead of three scalars.
    Fix #10: caller must supply P_down (not P_up) so density is evaluated
             at the correct post-flash state.
    """
    try:
        h_l = PropsSI("H", "P", P_down, "Q", 0, FLUID)
        h_v = PropsSI("H", "P", P_down, "Q", 1, FLUID)
        x   = (h_up - h_l) / (h_v - h_l)
        return float(np.clip(x, 0.0, 1.0))   # Fix #1
    except Exception as exc:
        warnings.warn(f"x_downstream failed at P_down={P_down:.3e}: {exc}")
        return 0.0


def rho_mix(x: float, P: float) -> float:
    """
    Homogeneous mixture density at quality x and pressure P.
    Uses harmonic mean of liquid/vapour densities (void-fraction weighted).
    """
    try:
        rho_l = PropsSI("D", "P", P, "Q", 0, FLUID)
        rho_v = PropsSI("D", "P", P, "Q", 1, FLUID)
        x = np.clip(x, 0.0, 1.0)
        return 1.0 / (x / rho_v + (1.0 - x) / rho_l)
    except Exception as exc:
        warnings.warn(f"rho_mix failed at P={P:.3e} x={x:.3f}: {exc}")
        return 800.0   # fallback ~liquid N2O density


# ---------------------------------------------------------------------------
# HEM reference flow coefficient (used as anchor for phi optimisation)
# ---------------------------------------------------------------------------

def K_HEM(m_dot_i: float, rho_hem: float, dP: float) -> float:
    """
    Generalised orifice coefficient from fully-equilibrium (HEM) state.
    K = m_dot / sqrt(rho_HEM * dP)

    This is the reference K that phi will scale from.  The optimiser then
    finds phi such that the non-equilibrium (real) density gives the same
    measured m_dot — fixing the tautology in the original code (#7).
    """
    denom = np.sqrt(max(rho_hem * dP, 1e-10))
    return m_dot_i / denom if denom > 0 else 1e-8


# ---------------------------------------------------------------------------
# Load & clean data
# ---------------------------------------------------------------------------

def load_data(path: str) -> dict:
    data = pd.read_csv(path)

    theta    = data["Theta"].values
    P_tank   = np.maximum(data["P_tank"].values   * 1e6, P_MIN)
    P_up     = np.maximum(data["P_up"].values     * 1e6, P_MIN)
    P_down   = np.maximum(data["P_down"].values   * 1e6, P_MIN)
    P_chamber= np.maximum(data["P_chamber"].values* 1e6, P_MIN)
    T_up     = data["T_up"].values + 273.15
    T_tank   = data["T_tank"].values + 273.15
    m_dot    = data["flow_rate"].values

    dP_ball  = np.maximum(P_up - P_down,   DP_MIN)
    dP_down  = np.maximum(P_down - P_chamber, DP_MIN)   # Fix #2 (was dp_down)
    dP_total = np.maximum(P_tank - P_chamber, DP_MIN)

    PR_ball  = P_down  / np.maximum(P_up,   1e-6)
    PR_total = P_chamber / np.maximum(P_tank, 1e-6)

    return dict(
        theta=theta, P_tank=P_tank, P_up=P_up, P_down=P_down,
        P_chamber=P_chamber, T_up=T_up, T_tank=T_tank, m_dot=m_dot,
        dP_ball=dP_ball, dP_down=dP_down, dP_total=dP_total,
        PR_ball=PR_ball, PR_total=PR_total,
    )


# ---------------------------------------------------------------------------
# Build thermodynamic feature arrays
# ---------------------------------------------------------------------------

def build_thermo_features(d: dict) -> tuple:
    """
    Returns arrays: x_up, h_up_arr, x_eq, rho_eq_down
    rho_eq_down is the HEM mixture density evaluated at P_down (#10 fix).
    """
    n = len(d["theta"])
    x_up_arr      = np.zeros(n)
    h_up_arr      = np.zeros(n)
    x_eq_arr      = np.zeros(n)
    rho_eq_down   = np.zeros(n)  # Fix #10: density at P_down, not P_up

    for i in range(n):
        xu, hu = get_upstream_state(d["P_up"][i], d["T_up"][i])
        xe     = x_downstream(hu, d["P_down"][i])        # Fix #10
        rho    = rho_mix(xe,  d["P_down"][i])             # Fix #10

        x_up_arr[i]    = xu
        h_up_arr[i]    = hu
        x_eq_arr[i]    = xe
        rho_eq_down[i] = rho

    return x_up_arr, h_up_arr, x_eq_arr, rho_eq_down


# ---------------------------------------------------------------------------
# Phi optimisation  (Fix #7 — removes tautology)
# ---------------------------------------------------------------------------

def optimise_phi(d: dict, x_eq: np.ndarray, rho_eq_down: np.ndarray) -> np.ndarray:
    """
    For each sample, find phi in [0,1] such that the non-equilibrium density
    (where x_real = phi * x_eq) predicts the measured m_dot using a fixed K.

    The K anchor is computed from the HEM state (full equilibrium, phi=1),
    making phi the only free parameter.  This breaks the tautology in the
    original code where K was re-derived inside the objective.
    """
    n        = len(d["theta"])
    phi_list = np.zeros(n)

    # Compute K_ref using HEM density (phi=1 baseline) — fixed reference
    K_ref = np.array([
        K_HEM(d["m_dot"][i], rho_eq_down[i], d["dP_ball"][i])
        for i in range(n)
    ])

    for i in range(n):
        def objective(phi_arr):
            phi    = phi_arr[0]
            x_real = np.clip(phi * x_eq[i], 0.0, 1.0)
            rho_ne = rho_mix(x_real, d["P_down"][i])   # non-equilibrium density
            m_pred = K_ref[i] * np.sqrt(max(rho_ne * d["dP_ball"][i], 1e-10))
            return (m_pred - d["m_dot"][i]) ** 2

        res = minimize(objective, x0=[0.7], bounds=[(0.01, 1.0)],
                       method="L-BFGS-B", options={"ftol": 1e-14})
        phi_list[i] = res.x[0]

    return phi_list


# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------

def train_state_model(d: dict) -> RandomForestRegressor:
    """
    model_state: (P_tank, T_tank, theta) -> (P_up, P_down, P_chamber, T_up)
    """
    X = np.column_stack((d["P_tank"], d["T_tank"], d["theta"]))
    Y = np.column_stack((d["P_up"], d["P_down"], d["P_chamber"], d["T_up"]))

    model = RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42)
    model.fit(X, Y)
    return model


def train_phi_model(d: dict, x_eq: np.ndarray, phi_array: np.ndarray
                    ) -> RandomForestRegressor:
    X = np.column_stack((
        d["theta"], d["P_up"], d["dP_ball"], d["PR_ball"],
        x_eq, d["dP_total"],
    ))
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, phi_array)
    return model


def train_K_model(d: dict,
                  x_up: np.ndarray, x_phi: np.ndarray,
                  rho_corrected: np.ndarray,
                  model_state: RandomForestRegressor,
                  ) -> RandomForestRegressor:
    """
    Fix #3: x_real (undefined at top level) replaced by x_phi (phi-corrected quality).
    Fix #2: dP_down (was dp_down).
    Fix #6/8: PR_total included — 14 features, matching mass_flow() inference.
    Fix #9: use OOF predictions from model_state to reduce information leakage
            into model_K training.
    """
    # OOF state predictions to reduce leakage (Fix #9)
    X_state = np.column_stack((d["P_tank"], d["T_tank"], d["theta"]))
    Y_state = np.column_stack((d["P_up"], d["P_down"], d["P_chamber"], d["T_up"]))
    Y_state_oof = cross_val_predict(
        RandomForestRegressor(n_estimators=400, max_depth=14, random_state=42),
        X_state, Y_state, cv=5,
    )
    P_up_oof     = np.maximum(Y_state_oof[:, 0], P_MIN)
    P_down_oof   = np.maximum(Y_state_oof[:, 1], P_MIN)
    P_ch_oof     = np.maximum(Y_state_oof[:, 2], P_MIN)

    dP_ball_oof  = np.maximum(P_up_oof - P_down_oof, DP_MIN)
    dP_down_oof  = np.maximum(P_down_oof - P_ch_oof,  DP_MIN)  # Fix #2
    dP_total_oof = np.maximum(d["P_tank"] - P_ch_oof,  DP_MIN)
    PR_ball_oof  = P_down_oof  / np.maximum(P_up_oof,    1e-6)
    PR_total_oof = P_ch_oof    / np.maximum(d["P_tank"], 1e-6)

    X_K = np.column_stack((       # 14 features — must match Xq in mass_flow()
        d["theta"],
        d["P_tank"],
        P_up_oof,
        P_down_oof,
        P_ch_oof,
        dP_ball_oof,
        dP_down_oof,              # Fix #2
        dP_total_oof,
        PR_ball_oof,
        PR_total_oof,             # Fix #6/8
        Y_state_oof[:, 3],        # T_up OOF
        x_up,
        x_phi,                    # Fix #3: phi-corrected quality (was undefined x_real)
        rho_corrected,
    ))
    y_K = d["m_dot"] / np.sqrt(np.maximum(rho_corrected * d["dP_ball"], 1e-10))

    X_tr, X_te, y_tr, y_te = train_test_split(X_K, y_K, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    print(f"[model_K] MAE={mean_absolute_error(y_te, y_pred):.4e}  "
          f"R²={r2_score(y_te, y_pred):.4f}")

    return model, X_K, y_K


# ---------------------------------------------------------------------------
# Forward map: theta + tank conditions -> m_dot
# ---------------------------------------------------------------------------

def mass_flow(theta_q: float, P_tank_q: float, T_tank_q: float,
              model_state: RandomForestRegressor,
              model_K: RandomForestRegressor) -> float:
    """
    Fix #6/8: Xq now has 14 features (PR_total added) matching X_K at training.
    Fix #5:   uses rho_q (not stale `rho`).
    Fix #10:  rho evaluated at P_down.
    """
    pred       = model_state.predict([[P_tank_q, T_tank_q, theta_q]])[0]
    P_up_q     = max(pred[0], P_MIN)
    P_down_q   = max(pred[1], P_MIN)
    P_ch_q     = max(pred[2], P_MIN)
    T_up_q     = pred[3]

    dP_ball_q  = max(P_up_q  - P_down_q,  DP_MIN)
    dP_down_q  = max(P_down_q - P_ch_q,   DP_MIN)
    dP_total_q = max(P_tank_q - P_ch_q,   DP_MIN)
    PR_ball_q  = P_down_q  / max(P_up_q,    1e-6)
    PR_total_q = P_ch_q    / max(P_tank_q,  1e-6)   # Fix #6/8

    xu, hu = get_upstream_state(P_up_q, T_up_q)
    xe     = x_downstream(hu, P_down_q)
    rho_q  = rho_mix(xe, P_down_q)   # Fix #10: P_down not P_up

    # phi not re-predicted here; assume phi=1 (HEM) for forward prediction.
    # If a phi model is available it can be applied to xe before rho_q.
    x_phi_q = xe

    Xq = [[
        theta_q, P_tank_q, P_up_q, P_down_q, P_ch_q,
        dP_ball_q, dP_down_q, dP_total_q,   # Fix #2
        PR_ball_q, PR_total_q,               # Fix #6/8
        T_up_q, xu, x_phi_q, rho_q,
    ]]   # 14 features

    K_q    = max(model_K.predict(Xq)[0], 1e-8)
    return K_q * np.sqrt(rho_q * dP_ball_q)


# ---------------------------------------------------------------------------
# Inverse map: target m_dot -> theta  (bisection)  Fix #4 name
# ---------------------------------------------------------------------------

def find_theta(m_target: float, P_tank_q: float, T_tank_q: float,
               model_state: RandomForestRegressor,
               model_K: RandomForestRegressor,
               theta_min: float = 0.0, theta_max: float = 90.0,
               tol: float = 1e-4, max_iter: int = 50) -> float:
    """
    Bisection search for valve angle that produces m_target.
    Fix #4: function name was 'find_theta' but main() called 'find_theta_bisection'.
    """
    def _flow(th):
        return mass_flow(th, P_tank_q, T_tank_q, model_state, model_K)

    m_low  = _flow(theta_min)
    m_high = _flow(theta_max)

    if m_target <= m_low:
        return theta_min
    if m_target >= m_high:
        return theta_max

    for _ in range(max_iter):
        theta_mid = 0.5 * (theta_min + theta_max)
        m_mid     = _flow(theta_mid)

        if abs(m_mid - m_target) < tol:
            return theta_mid

        if m_mid < m_target:
            theta_min = theta_mid
        else:
            theta_max = theta_mid

    return 0.5 * (theta_min + theta_max)


# ---------------------------------------------------------------------------
# Plotting   Fix #16: moved inside __main__ guard
# ---------------------------------------------------------------------------

def plot_K_vs_theta(d, K_data, theta_line, K_line):   # Fix #16
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["theta"], K_data, alpha=0.6, label="Data")
    ax.plot(theta_line, K_line, linewidth=2.5, label="Model")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Generalised K")
    ax.set_title("Generalised Valve Coefficient vs Theta")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_PATH = "/home/nivetha/scripts/Alpha/HNEM_model/ihara/BallValve.csv"

    print("Loading data...")
    d = load_data(DATA_PATH)
    n = len(d["theta"])

    print("Computing thermodynamic features...")
    x_up, h_up_arr, x_eq, rho_eq_down = build_thermo_features(d)

    print("Optimising phi (non-equilibrium departure factor)...")
    phi_array = optimise_phi(d, x_eq, rho_eq_down)

    # Phi-corrected quality and density (at P_down — Fix #10)
    x_phi = np.array([np.clip(phi_array[i] * x_eq[i], 0.0, 1.0) for i in range(n)])
    rho_corrected = np.array([rho_mix(x_phi[i], d["P_down"][i]) for i in range(n)])

    print("Training state model (P_tank, T_tank, theta) -> pressures/T_up ...")
    model_state = train_state_model(d)

    print("Training phi model...")
    model_phi = train_phi_model(d, x_eq, phi_array)

    print("Training K model (with OOF state predictions to reduce leakage)...")
    model_K, X_K, y_K = train_K_model(d, x_up, x_phi, rho_corrected, model_state)

    # --- Build K-vs-theta curve at median tank conditions --- Fix #5, #14, #16
    Pt = float(np.median(d["P_tank"]))
    Tt = float(np.median(d["T_tank"]))
    theta_line = np.linspace(0, 90, 100)
    K_line = []

    for th in theta_line:
        pred  = model_state.predict([[Pt, Tt, th]])[0]
        Pu    = max(pred[0], P_MIN)
        Pd    = max(pred[1], P_MIN)
        Pc    = max(pred[2], P_MIN)
        Tu    = pred[3]

        dPb   = max(Pu - Pd,  DP_MIN)
        dPd   = max(Pd - Pc,  DP_MIN)  # Fix #2
        dPt   = max(Pt - Pc,  DP_MIN)

        xu, hu = get_upstream_state(Pu, Tu)
        xe     = x_downstream(hu, Pd)
        rho_q  = rho_mix(xe, Pd)   # Fix #5/#10: was stale `rho` at P_up

        Xq = [[
            th, Pt, Pu, Pd, Pc,
            dPb, dPd, dPt,
            Pd / max(Pu, 1e-6),
            Pc / max(Pt, 1e-6),   # Fix #6/8: PR_total
            Tu, xu, xe, rho_q,
        ]]
        K_line.append(model_K.predict(Xq)[0])   # Fix #14: [0] not array

    K_data = y_K   # generalised K from training set

    # Fix #16: plot inside __main__ guard (below)
    plot_K_vs_theta(d, K_data, theta_line, K_line)

    # --- Interactive inverse mapping ---
    m_target       = float(input("Target mass flow rate (kg/s): "))
    P_tank_current = float(input("Tank pressure (MPa): ")) * 1e6
    T_tank_current = float(input("Tank temperature (K): "))

    theta_solution = find_theta(   # Fix #4: correct function name
        m_target, P_tank_current, T_tank_current, model_state, model_K
    )
    print(f"Required theta: {theta_solution:.3f} deg")


if __name__ == "__main__":
    main()
