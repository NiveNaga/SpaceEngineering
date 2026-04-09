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


   def hybrid_rhs(t, y, inputs, params, sim_options, t_erossion, cstar_ideal, R_obs):
       R_port, R_t = y[0], y[1]
       i_mdot_ox, i_Pc_exp = inputs
       Pc = float(i_Pc_exp(t))
       mdot_ox = float(i_mdot_ox(t))

       # 1. Fuel Regression
       
       Gox = mdot_ox / (np.pi * R_port**2)
       safe_Pc = max(Pc, 1e-6)
       k_smooth = 50 
       erosion_switch = 0.5 * (np.tanh(k_smooth * (t - t_erossion)) + 1)
       # print(f"a: {params['a']}, Gox: {Gox}, n: {params['n']}, safe_Pc: {safe_Pc}, m: {params['m']}, R_port: {R_port}, mdot_ox: {mdot_ox}")
       rdot = (params['a'] * (Gox**params['n']) * (safe_Pc**params['m']))
       # 2. Erosion
       mdot_f = sim_options['rho_fuel'] * rdot * (2 * np.pi * R_port * sim_options['L'])
       mdot_tot = mdot_ox + mdot_f
       safe_mdot = max(mdot_tot, 1e-9)
       # ─────────────────────────────
       # Throat erosion
       # ─────────────────────────────
       ke = params["k"]
       p  = params["p"]
       q  = params["q"]
       
       # 3. Calculate rdot_t
       rdot_t = (ke * (safe_mdot / (np.pi * R_t**2))**p * safe_Pc**q) * erosion_switch
       if t <= t_erossion:
         rdot_t = 0
       dR_t_dt = rdot_t
    
       return [rdot, dR_t_dt]

   def simulate_coupled(t_grid, inputs, R0, R_obs, Rt0,
                     params,t_erossion, cstar_ideal,
                     sim_options=None,
                     t_eval=None):
       if t_eval is None:
         t_eval = t_grid
       sol = solve_ivp(fun=lambda t, y: hybrid_rhs(t, y, inputs, params, sim_options, t_erossion, cstar_ideal, R_obs),
                    t_span=(t_grid[0], t_grid[-1]), y0= [R0, Rt0], t_eval=t_eval,
                    method='RK45', max_step = 0.5, rtol=1e-6, atol=1e-9)
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

   def automated_calibration(data_paths, R0_init, Rt0_init, R_final_meas, Rt_final_meas, t_erosion, cstar_ideal, sim_options):
    """
       Finds the global correction factors that make the 10-test sequence 
       land exactly on the final measured hardware dimensions.
    """
    low_b = [-13.8, 0.1, -0.2, 1e-12, 0.1, 0.0]
    high_b = [-4.6,  0.9,  0.5, 1e-4,  2.0, 1.5]
    params = np.array([-13.8, 0.4, 0.0, 1e-9, 0.3, 0.0])
    def objective_function(params, R0_init, Rt0_init, R_final_meas, Rt_final_meas, t_erosion, cstar_ideal, sim_options):
        ln_a, n, m, k, p, q = params
        a = np.exp(ln_a)
        params_sim = {'a': a, 'n': n, 'm': m, 'k': k, 'p': p, 'q': q}
        current_R = R0_init
        current_Rt = Rt0_init
        history = []
        data_path = np.array(data_paths)
        # Sequential Inner Loop
        for i in range(0, len(data_path)):
            df = pd.read_csv(data_path[i])
            # print(data_path[i])
            # Optimize a, n, m for this test
            # --- 1. Prepare Data ---
            t = df.iloc[:, 10].values
            Pc_exp = df.iloc[:, 12].values # Experimental Pressure (Pa)
            mdot_ox = df.iloc[:, 15].values
            
            # Create interpolants for this specific test
            i_md, i_Pc, _ = make_interpolants(t, mdot_ox, Pc_exp, df.iloc[:, 11].values)
            inputs = (i_md, i_Pc)
            sim = simulate_coupled(t, inputs, current_R, R_final_meas, current_Rt, params_sim, t_erosion[i],cstar_ideal[i],
                                sim_options=sim_options, t_eval=t)
            
            # Pass final state of this test to the next one
            current_R = sim['R_port'][-1]
            current_Rt = sim['Rt_throat'][-1]
            # print(f'current_R: {current_R}, current_Rt: {current_Rt}')
            history.append(params)

        # Calculate Global Error
        err_port = (current_R - R_final_meas) / R_final_meas
        err_throat = (current_Rt - Rt_final_meas) / Rt_final_meas
        
        print(f"Errors: Port={err_port*100:.2f}%, Throat={err_throat*100:.2f}%")
        return [err_port, err_throat]

    # Run the Outer Optimizer (Least Squares on the scaling factors)
    # Start with 1.0 (no correction)
    res = least_squares(objective_function, x0=params, bounds=(low_b, high_b), args=(R0_init, Rt0_init, R_final_meas, Rt_final_meas, t_erosion, cstar_ideal, sim_options),
          x_scale='jac',     # Crucial for parameters of different magnitudes
          diff_step=0.01,    # Larger step to ensure n and m actually move
          ftol=1e-10, 
          xtol=1e-10,
          method='trf')
    fit_params = {'a': np.exp(res.x[0]), 'n': res.x[1], 'm': res.x[2], 'k': res.x[3], 'p': res.x[4], 'q': res.x[5]}


    return res, fit_params # Returns the final [S_fuel, S_erosion]

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
       plot_simulation(t, cstar, 'C* across Burn Time', 'Time (secs)', 'C* (m/s)', 'r')
       plot_simulation(t, eta_cstar, 'c* efficiency across Burn Time', 'Time (secs)', 'η_c*', 'r')
       plot_simulation(t, OF, 'O/F across Burn Time', 'Time (secs)', 'O/F', 'g')
       plot_simulation(t, safe_mdot, 'Flow rate across Burn Time', 'Time (secs)', 'mdot', 'g')
       plot_simulation(t, R_t,  'Throat radius evolution over burn time', 'Time (secs)', 'R_throat m', 'g')
       plot_simulation(R_t, cstar, 'C* wrt Rt', 'R_throat m', 'C* (m/s)', 'k')
       plot_simulation(m_unique, c_unique, 'C* wrt mdot', 'Total mass flow rate (Kg/s)', 'C* (m/s)', 'k')
       plot_simulation(t, rdot, 'Regression rate wrt time', 'Time (secs)', 'rdot (m/s)', 'k')
       print(f'OF_time_avg: {OF_time_avg}, cstar_time_avg: {cstar_time_avg}, eta_cstar_avg: {eta_cstar_avg}')
       plot_simulation(t, safe_Pc,'Chamber Pressure across Time', 'Time (secs)', 'Pc (Mpa)', 'r')
       return cstar_time_avg, eta_cstar_avg, OF_time_avg

   # -------------------------
   # Fetch Inputs
   # -------------------------
   def main():
      dir_inp = input("Paste path to project folder:  ")
      all_data = input("Paste path to concat file: ")
      test = Path(dir_inp)
      test_path = test.as_posix()
      data_path = glob.glob(f"{test_path}/*.csv")
      data_path.sort()
      R0_init = 0.013       # initial port radius (15 mm)
      Rf = 0.052      #Fuel outer radius
      Rt0_init = 0.004     # initial throat radius (11 mm)
      sim_options = {'cstar': 500.0, 'Cf_func': 1.75, 'rho_fuel': 941.0, 'L': 0.1152, 'history':[], "log" : False}

      m_fi = float(input(f"Initial mass of the fuel in Kg: "))
      m_ff = float(input(f"Final mass of the fuel in Kg: "))
      t_erossion = list(map(float, input(f"Erossion begin time in secs: ").split(",")))
      cstar_ideal = list(map(float, input(f"Ideal cstar in m/s: ").split(",")))
      overall_cstar_ideal = float(input(f"overall Ideal cstar in m/s: "))
      Rt_obs = float(input(f"Final Throat radius of exp in (m):"))
      m_f = m_fi - m_ff
      vol = m_f/sim_options['rho_fuel']
      R_obs = np.sqrt(R0_init**2+(vol/(np.pi*sim_options['L'])))
      if len(data_path) == 0:
            print("No test data file found, exiting.")
            return
      elif len(data_path) == 1:
            print(f"found test data file: {data_path[0]}")
            print(f"Starting to process.")
            result, fit_params = automated_calibration(data_path, R0_init, Rt0_init, R_obs, Rt_obs, t_erossion, cstar_ideal, sim_options)
            
            df = pd.read_csv(data_path[0])
            t = df.iloc[:, 10].values
            F_exp = df.iloc[:, 11].values;
            pc = df.iloc[:, 12].values
            mdot_ox = df.iloc[:, 15].values
            i_md, i_Pc, i_F = make_interpolants(t, mdot_ox, pc, F_exp)
            inputs = (i_md, i_Pc)
            final_sim = simulate_coupled(t, inputs, R0_init, R_obs, Rt0_init, fit_params, np.mean(t_erossion),np.mean(cstar_ideal),
                                sim_options=sim_options, t_eval=t)
            post_proc = post_process(final_sim, inputs, fit_params, sim_options, np.mean(cstar_ideal), np.mean(t_erossion))
      else:
            print(f"found {len(data_path)} test data files. \n " \
                  "Enabling batch process:")
            result, fit_params= automated_calibration(data_path, R0_init, Rt0_init, R_obs, Rt_obs, t_erossion, cstar_ideal, sim_options)
            all = Path(all_data)
            all_path = all.as_posix()
            final_data = glob.glob(f"{all_path}/*.csv")
            df = pd.read_csv(final_data[0])
            t = df.iloc[:, 10].values
            F_exp = df.iloc[:, 11].values;\
            pc = df.iloc[:, 12].values
            mdot_ox = df.iloc[:, 15].values
            i_md, i_Pc, i_F = make_interpolants(t, mdot_ox, pc, F_exp)
            inputs = (i_md, i_Pc)
            final_sim = simulate_coupled(t, inputs, R0_init, R_obs, Rt0_init, fit_params, np.mean(t_erossion),np.mean(cstar_ideal),
                                sim_options=sim_options, t_eval=t)
            post_proc = post_process(final_sim, inputs, fit_params, sim_options, np.mean(cstar_ideal), np.mean(t_erossion))
      #print(result.x, sim)
      print(f"fit_params: {fit_params}")
except Exception as e:
   print(f"There was an error during the processing. The below exception arised \n {e}")
if __name__ == "__main__":
   main()
