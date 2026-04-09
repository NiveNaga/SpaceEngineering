##Author: NIVETHA RAJKUMAR
# Generalized solver for CG shift, slosh, and Monte-Carlo evaluation
# - Models: fixed structural masses + 4 tanks with slosh (simple pendulum/sinusoid)
# - Computes: time-varying CG, max perpendicular offset to thrust line, torque
# - Monte-Carlo: samples tank fill errors and slosh parameters, reports statistics
#
# Usage: change `params` dictionary or call the functions from another script.
# Outputs: summary stats printed, DataFrame displayed, two matplotlib plots (histogram and CDF).
#

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#from caas_jupyter_tools import display_dataframe_to_user

#np.random.seed(0)  # reproducible example
def compute_cg(struct_masses, tank_masses, tank_positions):
    """
    Compute center of mass given:
      - struct_masses: list of tuples (mass, position array-like)
      - tank_masses: list of masses for tanks (length 4)
      - tank_positions: list of position vectors for tanks (length 4)
    Returns total_mass, r_cg (3-vector)
    """
    total_mass = 0.0
    weighted = np.zeros(3)
    for m, pos in struct_masses:
        total_mass += m
        weighted += m * np.array(pos)
    for m, pos in zip(tank_masses, tank_positions):
        total_mass += m
        weighted += m * np.array(pos)
    r_cg = weighted / total_mass
    #print(weighted, m, struct_masses,total_mass, r_cg)
    return total_mass, r_cg

def simulate_time_varying_cg(struct_masses, tank_masses, tank_positions,
                             slosh_fraction_per_tank, slosh_amp_m, slosh_freq_hz,
                             slosh_phase, burn_dt, burn_time):
    """
    Simulate time history of CG during a burn with simple slosh model.
    Slosh model: a fraction of each tank mass (m_slosh = slosh_fraction_per_tank * tank_mass)
    oscillates laterally (perpendicular to thrust axis) with amplitude slosh_amp_m (m)
    and sinusoidal motion at slosh_freq_hz and phase slosh_phase (per tank).
    The slosh displacement vector is defined in the local tank transverse direction (x-y plane).
    Returns times (1D) and r_cg_history (Nt x 3)
    """
    times = np.arange(0, burn_time + burn_dt/2, burn_dt)
    Nt = len(times)
    r_cg_hist = np.zeros((Nt, 3))
    drain_fraction = np.zeros((Nt, 4))
    mdot_total = 0.033
    n_tanks = len(tank_positions)
    # base tank masses (liquid portion without slosh modal displacement)
    tank_base_masses = np.array(tank_masses) * (1 - slosh_fraction_per_tank)
    tank_slosh_masses = np.array(tank_masses) * slosh_fraction_per_tank
    
    # Precompute static added masses (base)
    total_base_mass, r_base = compute_cg(struct_masses, tank_base_masses, tank_positions)
    
    for i, t in enumerate(times):
        base_mass = np.abs(tank_base_masses) < 1.21
        drain_frac = np.clip(np.random.rand(n_tanks), 0.4, 1.0)
        if (np.abs(tank_base_masses) < 1.21).any():
          for i in range(n_tanks):
            if base_mass[i]:
              drain_frac[i] = 0
        flow = np.sum(drain_frac)
        drain_share = drain_frac * (mdot_total/flow)*burn_dt
        tank_base_masses -= drain_share
        # compute instantaneous lateral offset for each tank's slosh mass
        # assume slosh moves in a direction perpendicular to thrust (choose y-axis in body frame)
        slosh_offsets = slosh_amp_m * np.sin(2*np.pi*slosh_freq_hz * t + slosh_phase)
        # each tank slosh displacement vector (we'll place it in +y for tank index odd, -y for even just for diversity)
        displacements = []
        for j, pos in enumerate(tank_positions):
            #sign = 1 if (j % 2 == 0) else -1
            angle = np.random.uniform(0, 2*np.pi)
            disp = slosh_offsets[j] * np.array([np.cos(angle), np.sin(angle), 0.0])
            #disp = np.array([0.0, sign * slosh_offsets[j], 0.0])
            displacements.append(disp)
        # form instantaneous tank masses: base mass at tank nominal position + slosh mass at (tank position + disp)
        total_mass = 0.0
        weighted = np.zeros(3)
        for m, pos in struct_masses:
            total_mass += m
            weighted += m * np.array(pos)
        for j, pos in enumerate(tank_positions):
            base_m = tank_base_masses[j]
            slosh_m = tank_slosh_masses[j]
            total_mass += base_m + slosh_m
            weighted += base_m * np.array(pos) + slosh_m * (np.array(pos) + displacements[j])
        r_cg_hist[i, :] = weighted / total_mass
        drain_fraction[i, :] = drain_share
        #print(tank_base_masses, r_cg_hist[i, :], total_mass, t, slosh_fraction_per_tank)
        #print(total_base_mass, drain_share, total_mass, weighted, r_cg_hist[i, :], t, disp)
    return times, r_cg_hist, drain_fraction

def sample_and_evaluate(params):
    """
    Run a Monte-Carlo sampling of fill uncertainties and slosh parameters.
    Returns a pandas DataFrame with sample results and some summary statistics.
    """
    nsamples = params['nsamples']
    burn_time = params['burn_time']
    thrust_vector = np.array(params['thrust_vector'])
    a_max = params['acceleration']  # m/s^2 magnitude
    tau_avail = np.array(params['tau_avail'])
    margin_k = params['margin_k']
    theta_max_rad = np.array(params['theta_max_rad'] )
    I_axis = np.array(params['I_axis'])  # scalar inertia about axis of interest
    
    # unpack model
    struct_masses = params['struct_masses']  # list of (mass, pos)
    tank_positions = params['tank_positions']  # list of 4 pos vectors
    tank_nominal = np.array(params['tank_nominal_masses'])  # nominal masses of the 4 tanks
    tank_dry = np.array(params['tank_dry'])
    # results storage
    rows = []
    thrust_magnitude = 100
    
    for s in range(nsamples):
        # sample fill error: percent error per tank (normal distribution, mean 0)
        prop_mass = np.clip(np.random.rand(4), 0.4, 1.0)
        total_prop = np.sum(prop_mass)
        tank_prop_mass = prop_mass * (5.5/total_prop)
        #print(tank_prop_mass)
        #fill_fraction_errors = np.random.normal(loc=0.0, scale=params['fill_frac_std'], size=4)
        tank_masses = tank_prop_mass + tank_dry
        # slosh fraction sample (fraction of tank mass participating) per tank
        slosh_frac = np.clip(np.random.normal(loc=params['slosh_frac_mean'], scale=params['slosh_frac_std'], size=4), 0.05, 0.25)
        # slosh amplitude (m) per tank (positive)
        slosh_amp = np.abs(np.random.normal(loc=params['slosh_amp_mean'], scale=params['slosh_amp_std'], size=4))
        # slosh frequency (Hz) per tank
        slosh_freq = np.abs(np.random.normal(loc=params['slosh_freq_mean'], scale=params['slosh_freq_std'], size=4))
        # slosh phase
        slosh_phase = np.random.uniform(0, 2*np.pi, size=4)
        # Simulate time-varying CG during burn
        times, r_cg_hist, drain_frac = simulate_time_varying_cg(
            struct_masses, tank_masses, tank_positions,
            slosh_frac, slosh_amp, slosh_freq, slosh_phase,
            burn_dt=params['burn_dt'], burn_time=burn_time
        )
        cg[s,:]= r_cg_hist[:,:]
        '''r_cg_y[s,:]= r_cg_hist[:,1]
        r_cg_z[s,:]= r_cg_hist[:,2]'''
        drain[s,:]= drain_frac[:,:]
        drain_1[s,:]= drain_frac[:,0]
        drain_2[s,:]= drain_frac[:,1]
        drain_3[s,:]= drain_frac[:,2]
        drain_4[s,:]= drain_frac[:,3]
        # baseline CG (time=0) and max displacement during burn
        total_mass, r_cg0 = compute_cg(struct_masses, tank_masses, tank_positions)
        # compute displacement relative to nominal thrust line: assume nominal thrust line passes through r_cg0 projected along thrust_vector
        # perpendicular distance d_perp is magnitude of component of (r_cg - r_thrust_line_point) perpendicular to thrust vector.
        # We calculate for each time the perpendicular offset relative to initial CG projected onto plane normal to thrust.
        d_perps, h_axial = [[] for _ in range(2)]
        n = len(times)
        omega = np.zeros((n+1, 3))
        theta = np.zeros((n+1, 3))
        count=0
        tau_allowed = margin_k * tau_avail
        momentum = np.array([0.0, 0.0, 0.0])
        for rcg in r_cg_hist:
            delta = rcg - r_cg0
            dx, dy, dz = delta[0], delta[1], delta[2]
            axial_shift = delta @ thrust_vector
            perp_shift = np.linalg.norm(delta - (axial_shift * thrust_vector))
            # perpendicular component = delta - (delta dot t_hat) * t_hat
            #perp = delta - np.dot(delta, thrust_vector) * thrust_vector
            d_perps.append(perp_shift)
            h_axial.append(np.linalg.norm(axial_shift))
            tau_req = np.cross(delta, thrust_magnitude * thrust_vector)
            domega =  np.linalg.inv(np.diag(I_axis)) @ (tau_req - np.cross(omega[count], np.diag(I_axis) @ omega[count]))
            omega[count+1] = omega[count]+ (domega * params['burn_dt'])
            theta[count+1] = theta[count] + omega[count+1] * params['burn_dt']
            count += 1
            momentum += tau_req*params['burn_dt']
        max_x = np.abs(r_cg0[0])
        max_y = np.abs(r_cg0[1])
        max_h = np.abs(r_cg0[2])
        max_cg = np.array([max_x, max_y, max_h])
        # required torque magnitude = total_mass * a_max * max_d  (force = m*a, lever arm = max_d)
        #tau_req = total_mass * a_max * max_cg
        # check actuator authority
        authority_ok = tau_req <= tau_allowed
        # pointing constraint using theta_max and burn_time (approx small-angle)
        pointing_ok = theta[-1] <= theta_max_rad
        rows.append({
            'sample': s,
            'total_mass': total_mass,
            'max_x_m': max_x,
            'max_y_m': max_y,
            'max_h_m': max_h,
            'tau_req_x': tau_req[0],
            'tau_req_y': tau_req[1],
            'tau_allowed_Nm': tau_allowed,
            'authority_ok': authority_ok,
            'delta_theta_x': theta[-1,0],
            'delta_theta_y': theta[-1,1],
            'delta_theta_z': theta[-1,2],
            'pointing_ok': pointing_ok,
            'Tank_fill_frac_range': np.max(tank_prop_mass) - np.min(tank_prop_mass),
            'Tank_fill_frac_max': np.max(tank_prop_mass),
            'Tank_fill_frac': tank_prop_mass,
            'mean_slosh_frac': slosh_frac.mean(),
            'momentum_x': momentum[0],
            'momentum_y': momentum[1],
            'momentum_z': momentum[2],
        })
    df = pd.DataFrame(rows)
    # summary
    summary = {
        'nsamples': nsamples,
        'frac_authority_ok': df['authority_ok'].mean(),
        'frac_pointing_ok': df['pointing_ok'].mean(),
        'tau_req_x_95pct': df['tau_req_x'].quantile(1.0),
        'tau_req_y_95pct': df['tau_req_y'].quantile(1.0),
        'Pointing_error_x': df['delta_theta_x'].max(),
        'Pointing_error_y': df['delta_theta_y'].max(),
        'Pointing_error_z': df['delta_theta_z'].max(),
        'Tank_fill_frac_max': df['Tank_fill_frac_max'].quantile(1.0),
        #'tau_req_max': df['tau_req_x'].max(),
        'max_d_95pct_m': df['max_y_m'].quantile(1.0),
        'max_x_95pct_m': df['max_x_m'].quantile(1.0),
        'max_h_95pct_m': df['max_h_m'].quantile(1.0),
    }
    return df, summary

# Default parameters (user can edit)
params = {
    'nsamples': 2000,
    'burn_time': 100.0,     # seconds
    'burn_dt': 0.1,      # timestep for slosh sim
    'thrust_vector': [0, 0, 1],  # +Z body axis (thrust along +Z)
    'acceleration': 0.5,  # m/s^2 (F/m nominal linear acceleration)
    'tau_avail': [0.2, 0.2, 0.2],   # N*m available control torque
    'margin_k': 0.85,      # fraction of tau_avail allowed to be used
    'theta_max_rad': [0.009, 0.009, 0.009], # allowable pointing error in rad
    'I_axis': [32.5, 32.5, 19.75],     # kg*m^2 about axis of interest
    
    # structural masses: list of (mass, [x,y,z])
    'struct_masses': [
        (265.0, [0.0, 0.0, 0.0]),  # spacecraft dry mass at origin
    ],
    # tanks: positions and nominal masses (4 tanks)
    'tank_positions': [
        [0.16,  0.16, -0.65],
        [-0.16, 0.16, -0.65],
        [0.16, -0.16, -0.65],
        [-0.16,-0.16, -0.65]
    ],
    'tank_nominal_masses': [2.5505, 2.5505, 2.5505, 2.5505],  # kg each
    'tank_dry' : [1.2, 1.2, 1.2, 1.2],
    # sampling parameters for uncertainties/slosh
    #'fill_frac_std': 0.15,  # 15% std dev in fill fraction per tank
    'slosh_frac_mean': 0.08, # mean fraction of tank mass participating in slosh
    'slosh_frac_std': 0.04,
    'slosh_amp_mean': 0.05,  # mean slosh amplitude (m)
    'slosh_amp_std': 0.02,
    'slosh_freq_mean': 0.5,  # Hz natural frequency
    'slosh_freq_std': 0.2,
}
t = np.arange(0, params['burn_time'] + params['burn_dt']/2, params['burn_dt'])
cg = np.zeros((params['nsamples'], len(t), 3))
r_cg_x = np.zeros((params['nsamples'], len(t)))
r_cg_y = np.zeros((params['nsamples'], len(t)))
r_cg_z = np.zeros((params['nsamples'], len(t)))
drain = np.zeros((params['nsamples'], len(t), 4))
drain_1 = np.zeros((params['nsamples'], len(t)))
drain_2 = np.zeros((params['nsamples'], len(t)))
drain_3 = np.zeros((params['nsamples'], len(t)))
drain_4 = np.zeros((params['nsamples'], len(t)))
# Run Monte-Carlo
df, summary = sample_and_evaluate(params)
# Print summary and show DataFrame and plots
print("Monte-Carlo summary (n={}):".format(summary['nsamples']))
print()
for k,v in summary.items():
    if k!='nsamples':
        print(f"  {k}: {v}")

# Save results to CSV for user download
outpath = "/home/nivetha/scripts/output/Alpha_cg_slosh.csv"
df.to_csv(outpath, index=False)
print(f"\nSaved detailed results to: {outpath}")

# Display a short table of worst 10 cases sorted by required torque
df_sorted = df.sort_values('tau_req_x', ascending=False).head(100).reset_index(drop=True)
print(df_sorted)

import seaborn as sns

X_flat = cg.reshape(-1, 3)
Y_flat = drain.reshape(-1, 4)
corr = np.corrcoef(np.hstack((X_flat, Y_flat)).T)

df_corr = pd.DataFrame(corr,
    index=[f'cg{i}' for i in range(3)] + [f'drain{j}' for j in range(4)],
    columns=[f'cg{i}' for i in range(3)] + [f'drain{j}' for j in range(4)]
)

sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title("Drain vs CG")

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(drain_1)
print(len(labels))
plt.figure(figsize=(10,6))
for cluster in range(3):
    cluster_idx = np.where(labels == cluster)[0]
    for idx in cluster_idx[:5]:  # plot first 5 samples from this cluster
        plt.plot(drain_1[idx,:], cg[idx,:,0], alpha=0.5)
plt.xlabel('Drain kg')  # whatever units your drain has
plt.ylabel('CG X-shift')
plt.title('CG X-shift vs Drain for different drain patterns')

plt.show()

# ------------------------
# Plot Hexbin for Torque X vs Fill Imbalance
# ------------------------

'''
plt.figure(figsize=(8,5))
hb = plt.hexbin(df['tau_req_x'], df['Tank_fill_frac_range'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max Torque X (N·m)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Torque X vs Tank Fill Fraction Imbalance')
plt.grid(True)

plt.figure(figsize=(8,5))
hb = plt.hexbin(df['tau_req_y'], df['Tank_fill_frac_range'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max Torque Y (N·m)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Torque Y vs Tank Fill Fraction Imbalance')
plt.grid(True)


plt.figure(figsize=(8,5))
hb = plt.hexbin(df['momentum_x'], df['Tank_fill_frac_range'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Momentum X (Nms)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Momentum X vs Tank Fill Fraction Imbalance')
plt.grid(True)

plt.figure(figsize=(8,5))
hb = plt.hexbin(df['momentum_y'], df['Tank_fill_frac_range'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Momentum Y (Nms)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Momentum Y vs Tank Fill Fraction Imbalance')
plt.grid(True)

# ------------------------
plt.figure(figsize=(8,5))
hb = plt.hexbin(df['tau_req_x'], df['Tank_fill_frac_max'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max Torque X (N·m)')
plt.ylabel('Tank Fill Fraction Max (Kg)')
plt.title('Torque X vs Tank Fill Fraction Max')
plt.grid(True)

plt.figure(figsize=(8,5))
hb = plt.hexbin(df['tau_req_y'], df['Tank_fill_frac_max'], gridsize=40, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max Torque Y (N·m)')
plt.ylabel('Tank Fill Fraction Max (Kg)')
plt.title('Torque Y vs Tank Fill Fraction Max')
plt.grid(True)

# Plot empirical CDF of max_d (single plot)
sorted_d = np.sort(df['max_y_m'].values)
sorted_x = np.sort(df['max_x_m'].values)
sorted_h = np.sort(df['max_h_m'].values)
plt.figure(figsize=(7,4))
plt.scatter(df['max_y_m'], df['Tank_fill_frac_range'], label='CG offset Y (m)')
plt.scatter(df['max_x_m'], df['Tank_fill_frac_range'], label='CG offset X (m)')
plt.xlabel('Max Perpendicular CG offset (m)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('CG offset as fn(Tank Fill Fraction, sloshing)')
plt.legend()
plt.grid(True)

plt.figure(figsize=(7,4))
plt.plot(df['max_h_m'], df['Tank_fill_frac_range'], c = 'red')
plt.xlabel('Max axial CG offset (m)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('CG offset as fn(Tank Fill Fraction, sloshing)')
plt.grid(True)

plt.figure(figsize=(7,4))
hb = plt.hexbin(df['delta_theta_x'], df['Tank_fill_frac_range'], gridsize=40, cmap='inferno', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max deflection angle (rad)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Max deflection of SC along X axis wrt Tank Fill Fraction')
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
hb = plt.hexbin(df['delta_theta_y'], df['Tank_fill_frac_range'], gridsize=40, cmap='inferno', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max deflection angle (rad)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Max deflection of SC along Y axis wrt Tank Fill Fraction')
plt.grid(True)
plt.show()

plt.figure(figsize=(7,4))
hb = plt.hexbin(df['delta_theta_z'], df['Tank_fill_frac_range'], gridsize=40, cmap='inferno', mincnt=1)
plt.colorbar(hb, label='Sample Count')
plt.xlabel('Max deflection angle (rad)')
plt.ylabel('Tank Fill Fraction Spread (max - min) Kg')
plt.title('Max deflection of SC along Z axis wrt Tank Fill Fraction')
plt.grid(True)
plt.show()
'''
