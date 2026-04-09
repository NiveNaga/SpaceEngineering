from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import newton
import glob
g0 = 9.80665
try:
   def make_interpolants(t_grid, mdot_ox, Pc, Thrust):
       """Return lambda interpolators for inputs (safe for scalar/array t calls)."""
       def i_md(t): return np.interp(t, t_grid, mdot_ox)
       def i_Pc(t): return np.interp(t, t_grid, Pc)
       def i_F(t): return np.interp(t, t_grid, Thrust)
       return i_md, i_Pc, i_F

   # -------------------------
   # Coupled ODE: dR/dt, dRt/dt
   # -------------------------
   def hybrid_rhs(t, y, inputs, params, sim_options, t_erossion, cstar_ideal, R_obs):
       R_port, R_t = float(y[0]), float(y[1])
       # ─────────────────────────────
       # Unpack inputs
       # ─────────────────────────────
       i_mdot_ox, i_Pc, i_F = inputs
       Pc = float(i_Pc(t))
       F  = float(i_F(t))
       mdot_ox = float(i_mdot_ox(t))
       # ─────────────────────────────
       # Geometry
       # ─────────────────────────────
       A_t = np.pi * R_t**2
       A_port = np.pi * R_port**2
       L_fuel = sim_options["L"]
       A_burn = 2 * np.pi * R_port * L_fuel

       rho_f = sim_options["rho_fuel"]

       # ─────────────────────────────
       # Thrust coefficient (pure experiment)
       # ─────────────────────────────
       # Cf = F / (Pc* 1.0e+6 * A_t)

       # ─────────────────────────────
       # Regression model parameters
       # ─────────────────────────────
       a = params["a"]
       n = params["n"]
       m = params["m"]

       # ─────────────────────────────
       # Oxidizer mass flux
       # ─────────────────────────────
       Gox = mdot_ox / A_port
       # ─────────────────────────────
       # Implicit c* solve
       # ─────────────────────────────
       safe_Pc = max(Pc, 1e-9) 
       k_smooth = 50 
       erosion_switch = 0.5 * (np.tanh(k_smooth * (t - t_erossion)) + 1)

       rdot = a * (Gox**n) * (safe_Pc**m)
       mdot_f = rho_f * rdot * A_burn
       mdot_tot = mdot_ox + mdot_f
       safe_mdot = max(mdot_tot, 1e-9)
       '''def cstar_residual(cstar):
         residual = 0
         if mdot_tot > 0:
             residual = (Pc * A_t) / mdot_tot
         return cstar - residual

       # Initial guess: previous timestep or nominal
       cstar0 = sim_options.get("cstar", 1500.0)

       try:
         cstar = newton(
            cstar_residual,
            cstar0,
            tol=1e-6,
            maxiter=20
         )

       except RuntimeError:
         cstar = cstar0
       n_cstar = cstar/cstar_ideal
       # ─────────────────────────────
       # Final derived quantities
       # ─────────────────────────────
       if mdot_f > 0: 
         of = mdot_ox/mdot_f
       else:
         of = 0
       #Isp = Cf * cstar / g0
       print(f'(t: {t}, rdot: {rdot}, R_port: {R_port}, mdot_f: {mdot_f}, A_burn: {A_burn}')
       #print(f'(time:{t},Isp:{Isp}, F: {F}, mdot_f: {mdot_f}, cstar: {cstar}, Cf: {Cf}')'''
       # ─────────────────────────────
       # Throat erosion
       # ─────────────────────────────
       ke = params["k"]
       p  = params["p"]
       q  = params["q"]

       # 3. Calculate rdot_t
       rdot_t = (ke * (safe_mdot / A_t)**p * safe_Pc**q) * erosion_switch
       if t <= t_erossion:
         rdot_t = 0
       # ─────────────────────────────
       # ODEs
       # ─────────────────────────────
       dR_port_dt = rdot
       dR_t_dt    = rdot_t
       
       # Optional logging
       '''if sim_options.get("log", False):
          sim_options["history"].append(
            (t, Pc, of, cstar, n_cstar, mdot_ox, mdot_f, rdot, rdot_t)
          )'''
       return [dR_port_dt, dR_t_dt]       # Compute Cf and Ueq (we use current At)

   # -------------------------
   # Simulation wrapper
   # -------------------------
   def simulate_coupled(t_grid, inputs, R0, R_obs, Rt0,
                     params,t_erossion, cstar_ideal,
                     sim_options=None,
                     t_eval=None):
       """
       - R0, Rt0: initial radii (m)
       - a,n,m: regression parameters (floats)
       - erosion_params: dict with keys k_e,p,q (defaults provided)
       """

       # ─────────────────────────────
       # Initialization
       # ─────────────────────────────

       '''if sim_options is None:
         sim_options = {'cstar': 500.0, 'Cf_func': Cf_constant, 'rho_fuel': 941.0, 'L': 0.1152}'''
       # ─────────────────────────────
       # interpolants
       # ─────────────────────────────

       if t_eval is None:
         t_eval = t_grid
       # ─────────────────────────────
       # integrate solver
       # ─────────────────────────────
       sol = solve_ivp(fun=lambda t, y: hybrid_rhs(t, y, inputs, params, sim_options, t_erossion, cstar_ideal, R_obs),
                    t_span=(t_grid[0], t_grid[-1]), y0= [R0, Rt0], t_eval=t_eval,
                    method='RK45', max_step = 1, rtol=1e-6, atol=1e-9)

       if not sol.success:
         raise RuntimeError("ODE integration failed: " + str(sol.message))
       
       R_port = sol.y[0]
       Rt_throat = sol.y[1]
       At = np.pi * Rt_throat**2

       return {
        't': sol.t,
        'R_port': R_port,
        'Rt_throat': Rt_throat,
        'At': At
       }

   # -------------------------
   # Plot helper
   # -------------------------
   def plot_simulation(x, y, lbl, xaxis, yaxis, color):
       plt.figure(figsize=(8, 6))
       plt.plot(x, y, color = color)
       plt.title(lbl)
       plt.xlabel(xaxis)
       plt.ylabel(yaxis)
       plt.tight_layout()
       plt.show()

   # -------------------------
   # residual function optimizing [a, n, m, k, p, q]
   # -------------------------
   def residuals_logparams(x, t_grid, inputs, R0, Rt0, R_obs, Rt_obs, sim_options, t_erossion, cstar_ideal):
       ln_a, n, m, k, p, q = x
       a = np.exp(ln_a)
       params = {'a': a, 'n': n, 'm': m, 'k': k, 'p': p, 'q': q}
       try:
          sim = simulate_coupled(t_grid, inputs, R0, R_obs, Rt0, params,t_erossion,cstar_ideal,
                                sim_options=sim_options, t_eval=t_grid)
          err_Rf = (sim['R_port'][-1] - R_obs)/R_obs
          err_Rtf = (sim['Rt_throat'][-1] - Rt_obs)/Rt_obs
          return np.array([err_Rf, err_Rtf])
       except Exception:
          # Return a large error if the ODE fails for these params
          return np.array([1e3, 1e3])

   # -------------------------
   # Post-Process
   # -------------------------
   def post_process(sim, inputs, fit_params, sim_options, cstar_ideal, t_erossion):
       t = sim['t']
       R = sim['R_port']
       R_t = sim['Rt_throat']
       rho_f = sim_options["rho_fuel"]
       L = sim_options["L"]

       mdot_ox = np.array([inputs[0](ti) for ti in t])
       Pc = np.array([inputs[1](ti) for ti in t])      # MUST be Pa

       A_port = np.pi * R**2
       A_t = np.pi * R_t**2


       #safe_Pc = np.clip(np.nan_to_num(Pc, nan=1e-6), 1e-6, None)

       # Regression
       a, n, m = fit_params["a"], fit_params["n"], fit_params["m"]
       Gox = mdot_ox / A_port

       safe_Pc = np.clip(np.nan_to_num(Pc, nan=1e-5), 1e-5, None)

       rdot = a * (Gox**n) * (safe_Pc**m)
       # Fuel mass flow (diagnostic)
       A_burn = 2 * np.pi * R * L
       mdot_f = rho_f * rdot * A_burn
       # O/F ratio
       mdot_tot = mdot_ox + mdot_f
       unique_mdot = np.unique(mdot_tot)
       OF = np.where(mdot_f != 0, mdot_ox / mdot_f, 0)
       safe_mdot = np.clip(np.nan_to_num(mdot_tot, nan=1e-3), 1e-3, None)
       # erossion
       ke, p, q = fit_params["k"], fit_params["p"], fit_params["q"]
       # ---- c* (m/s) ----
       cstar = (safe_Pc * 1e6 * A_t) / safe_mdot     # m/s
       cstar = np.clip(np.nan_to_num(cstar, nan=0.1), 0.1, None)
       cstar_mean = np.array([cstar[mdot_tot==xi].mean() for xi in unique_mdot])
       # c* efficiency
       eta_cstar = cstar / cstar_ideal
       cstar_time_avg = np.mean(cstar)
       eta_cstar_avg = np.mean(eta_cstar)
       OF_time_avg = np.mean(OF)
       rdot_avg = np.mean(rdot)
       idx = np.argsort(t)
       t_sort = t[idx]
       c = cstar[idx]
       m = safe_mdot[idx]
       df = pd.DataFrame({"t": t_sort, "c": c, "m": safe_mdot})
       df = df.groupby("t", as_index=False).mean()
       t_unique = df["t"].values
       c_unique = df["c"].values
       m_unique = df["m"].values
       # plots
       #plot_simulation(t, cstar, 'C* across Burn Time', 'Time (secs)', 'C* (m/s)', 'r')
       #plot_simulation(t, eta_cstar, 'c* efficiency across Burn Time', 'Time (secs)', 'η_c*', 'r')
       plot_simulation(t, OF, 'O/F across Burn Time', 'Time (secs)', 'O/F', 'g')
       #plot_simulation(t, safe_mdot, 'Flow rate across Burn Time', 'Time (secs)', 'mdot', 'g')
       #plot_simulation(t, R_t,  'Throat radius evolution over burn time', 'Time (secs)', 'R_throat m', 'g')
       #plot_simulation(R_t, cstar, 'C* wrt Rt', 'R_throat m', 'C* (m/s)', 'k')
       #plot_simulation(m_unique, c_unique, 'C* wrt mdot', 'Total mass flow rate (Kg/s)', 'C* (m/s)', 'k')
       print(f'OF_time_avg: {OF_time_avg}, cstar_time_avg: {cstar_time_avg}, eta_cstar_avg: {eta_cstar_avg}, rdot_avg: {rdot_avg}')
       plot_simulation(t, safe_Pc,'Chamber Pressure across Time', 'Time (secs)', 'Pc (Mpa)', 'r')
       plot_simulation(t, rdot, 'rdot wrt Time', 'Time (secs)' ,'rdot (m/s)', 'b')
       plot_simulation(t, Gox, 'Ox mass flux along burn', 'Time (secs)', 'Gox (Kg/m2.s)', 'g')
       plot_simulation(t, R, 'Port radius evolution', 'Time (secs)', 'Port radius (m)', 'k')
       return cstar_time_avg, eta_cstar_avg, OF_time_avg
   # -------------------------
   # Evaluate Empirical data
   # -------------------------
   def processing(data, R0, Rt0, data_input):
       t = data.iloc[:, 10].values
       F_exp = data.iloc[:, 11].values
       pc = data.iloc[:, 12].values
       mdot_ox = data.iloc[:, 15].values

       # User Inputs
       i_md, i_Pc, i_F = make_interpolants(t, mdot_ox, pc, F_exp)
       inputs = (i_md, i_Pc, i_F)
       

       m_fi = float(input(f"Initial mass of fuel (kg): "))
       m_ff = float(input(f"Final mass of fuel (kg): "))
       t_erossion = float(input(f"Erosion start time (s): "))
       cstar_ideal = float(input(f"Ideal cstar (m/s): "))
       Rt_obs = float(input(f"Final Throat radius (m): "))
       sim_options = {'rho_fuel': 946.9642, 'L': 0.1152}
       vol_consumed = (m_fi - m_ff) / sim_options['rho_fuel']
       R_obs = np.sqrt(R0**2 + (vol_consumed / (np.pi * sim_options['L'])))

       # Initial Guesses: [ln(a), n, m, k, p, q]
       # ln(1e-6) is approx -13.8.
       x0 = np.array([-13.8, 0.5, 0.1, 1e-8, 0.5, 0.1])

       # Define Bounds for Log-transformed space
       # ln_a bounds: ln(1e-6) to ln(1e-2)
       # n bounds: 0.1 to 0.9 (Standard for hybrids)
       # m bounds: -0.2 to 0.5
       low_b = [-13.8, 0.1, -0.2, 1e-12, 0.1, 0.0]
       high_b = [-4.6,  0.9,  0.5, 1e-4,  2.0, 1.5]
       print("Optimizing parameters...")
       res = least_squares(
          fun=residuals_logparams, 
          x0=x0, 
          bounds=(low_b, high_b),
          args=(t, inputs, R0, Rt0, R_obs, Rt_obs, sim_options, t_erossion, cstar_ideal),
          x_scale='jac',     # Crucial for parameters of different magnitudes
          diff_step=0.01,    # Larger step to ensure n and m actually move
          ftol=1e-10, 
          xtol=1e-10,
          method='trf')
       # Convert ln(a) back to a
       ln_a_fit, n_fit, m_fit, k_fit, p_fit, q_fit = res.x
       a_fit = np.exp(ln_a_fit)
       fit_params = {'a': a_fit, 'n': n_fit, 'm': m_fit, 'k': k_fit, 'p': p_fit, 'q': q_fit}
       print(f"\n--- Optimization Results ---")
       print(f"a: {a_fit:.2e} | n: {n_fit:.4f} | m: {m_fit:.4f}")
       print(f"k: {k_fit:.2e} | p: {p_fit:.4f} | q: {q_fit:.4f}")
       # Final Simulation for plotting
       best_sim = simulate_coupled(t, inputs, R0, R_obs, Rt0, fit_params, 
                                t_erossion, cstar_ideal, sim_options=sim_options, t_eval=t)
       post_process(best_sim, inputs, fit_params, sim_options, cstar_ideal, t_erossion)
       return res, fit_params, best_sim
   # -------------------------
   # Fetch Inputs
   # -------------------------
   def main():
      dir_inp = input("Paste path to project folder:  ")
      test = Path(dir_inp)
      test_path = test.as_posix()
      data_path = glob.glob(f"{test_path}/*.csv")
      R0 = 0.04185   #0.013   # initial port radius (15 mm)
      #Rf = 0.053 #0.052      #Fuel outer radius
      Rt0 = 0.004     # initial throat radius (11 mm)
      if len(data_path) == 0:
            print("No test data file found, exiting.")
            return
      else len(data_path):
            if len(data_path) != 1:
               print('Multiple files found, proceed batch process for Reignition models')
               break
            print(f"found test data file: {data_path[0]}")
            print(f"Starting to process.")
            data_ = pd.read_csv(data_path[0])
            data_input = str(data_path[0])
            result, fit_params, sim = processing(data_, R0, Rt0, data_input)
except Exception as e:
   print(f"There was an error during the processing. The below exception arised \n {e}")
if __name__ == "__main__":
   main()
