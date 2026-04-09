import numpy as  np
import pandas as pd
import os, sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from types import SimpleNamespace
from itertools import combinations

# Constants
CONST_GRAVITATIONAL_ACC = 9.80665  # m/s^2
THRUST_FORCE = 22  # N
TORQUE_LIMIT = 20  # N·m per axis
FIRE_CYCLE_TIME = 0.01  # seconds per control cycle dt

# Initial parameters
initialMass = 995  # kg

# --- Constants ---
I_sp = 280  # s
rampup = 10
burn_time = 300
spin_duration = 30

MIB = 0.35  # N·s
DEADBAND = 0.002
initialAngularVelocityX = 0.00000000
initialAngularVelocityY = 0.00000000
initialAngularVelocityZ = 3.14159265
# Euler angles (3-1-2 sequence)
initialEulerAnglePhiX = 0  # rad
initialEulerAnglePhiY = 0
initialEulerAnglePhiZ = 0

# Inertial velocities [m/s]
initialInertialVelocityX = 0
initialInertialVelocityY = 0
initialInertialVelocityZ = 0

# Thrust and misalignment
F_max = 3850  # N
initialDistanceNozzleThroatCG = 0.75  # m
motorOffset = 0.01  # m
initialThrustMisalignment = np.deg2rad(3)  # rad
initialThrust = 0

#FinalParameters
hf = 1.0   #finalDistanceNozzleThroatCG

#Thrust misalignment: Parameter setting for sigmoid functions
sigmoidRiseTime = 4110 # Time to start a stepwise change in the sigmoid function
sigmoidPreValue = 0 # Value before the step
sigmoidPostValue= 1 # Value after the step
sigmoidSlope = 4 # Slope of the sigmoid function
thrustMisalignmentFactor = 4
sigmoidFactor = thrustMisalignmentFactor * np.pi / 180 #Thrust Misalignment factor in rad (0.122 rads)

Thetax, Thetay, Thetaz, torquex, torquey, torquez, time, wx_l, wy_l, wz_l, phix_l, phiy_l, phiz_l, V_x, V_y, V_z, mdot, rcs, rcs_x, rcs_y, rcs_z, m_rcs, fuel, eff, VPE, delta_Vx, delta_Vy, delta_Vz, delta_V, torques = [[] for _ in range(30)]
v = np.zeros(28)
# Each row: [Fx, Fy, Fz, px, py, pz]
thruster_config = np.array([
     [+1,  0,  0,  0,  1,  1],
     [-1,  0,  0,  0, -1,  1],
     [ 0, +1,  0,  1,  0,  1],
     [ 0, -1,  0, -1,  0,  1],
     [ 0,  0, +1,  1,  1,  0],
     [ 0,  0, -1, -1,  1,  0],
     [-1,  0,  0,  0,  1, -1],
     [0,  -1,  0,  1,  0, -1]])
# Define thruster pairs for each axis
thruster_pairs = {
     'x': (0, 1, 6),  # +X, -X, -X_redundant thrusters
     'y': (2, 3, 7),  # +Y, -Y, -Y_redundant thrusters
     'z': (4, 5)}
for t in thruster_config:
    F = t[0:3] * THRUST_FORCE
    r = t[3:6]
    torque = np.cross(r, F)
    torques.append(torque)
torques = np.array(torques)
'''
thruster_config = np.array([
    [+1,  0,  0,  0,  1,  1],
    [-1,  0,  0,  0, -1,  1],
    [ 0, +1,  0, -1,  0, -1],
    [ 0, -1,  0,  1,  0, -1],
    [ 0,  0, +1,  1,  1,  0],
    [ 0,  0, -1, -1, -1,  0],
])  # Each row: [Fx, Fy, Fz, px, py, pz]
'''
control_config = {
    'burn_start': 4100,  # e.g., burn starts at 4450s
    'kp': 2,
    'kd': 15,
    'Kp_g': 0.5,
    'Kd_g': 5}
I_dry = np.diag([198.75, 198.75, 78.125]) #np.diag([169.0897, 169.0897, 149.2034])
I_full = np.diag([316.119, 316.119, 124.375]) #np.diag([286.4938, 286.4938, 210.3429])
m_dry = 595  # kg
Mx = 0
My = 0
Mz = 0
#Load and Transform GMAT Ephemeris
from scipy.spatial.transform import Rotation as Rt
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

df = pd.read_csv('/mnt/c/Users/nivet/OTV/Orbital Mechanics/GMAT/report_spin.txt', delim_whitespace=' ')
df['time_sec'] = (df['OTV.UTCModJulian'] - df['OTV.UTCModJulian'].iloc[0]) * 86400.0
df = df.sort_values('time_sec')
times = df['time_sec'].values
quats = df[['OTV.Q1', 'OTV.Q2', 'OTV.Q3', 'OTV.Q4']].values
df['OTV.AngularVelocityZ'] = np.deg2rad(df['OTV.AngularVelocityZ'])
ang_vel = df[['OTV.AngularVelocityX', 'OTV.AngularVelocityY', 'OTV.AngularVelocityZ']].values
key_rots = Rt.from_quat(quats)
slerp = Slerp(times, key_rots)
t_uniform = np.arange(times[0], times[-1], 1.0)  # e.g., 1 second interval
#print(len(t_uniform))
interp_rots = slerp(t_uniform)
interp_quats = interp_rots.as_quat()
Qdes = interp_quats[:, [3, 0, 1, 2]].tolist()

wx_interp = interp1d(times, ang_vel[:, 0], kind='linear', fill_value='extrapolate')
wy_interp = interp1d(times, ang_vel[:, 1], kind='linear', fill_value='extrapolate')
wz_interp = interp1d(times, ang_vel[:, 2], kind='linear', fill_value='extrapolate')
#df1 = pd.DataFrame({'time_sec': t_uniform, 'Qdes': Qdes, 'AngVels': AngVels})
q_esti = Qdes[0]
q_esti0 = q_esti[0]
q_esti1 = q_esti[1]
q_esti2 = q_esti[2]
q_esti3 = q_esti[3]


def check_for_nans_or_infs(name, array):
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        raise ValueError(f"{name} contains NaNs or Infs:\n{array}")

total_delta_v = np.array([0.0, 0.0, 0.0])
last_time = t_uniform[0]
def step_callback(t, dcm, thrust_body, m):
    global total_delta_v, last_time
    dt = t - last_time
    delV = (dcm @ thrust_body) / m
    total_delta_v += delV * dt
    last_time = t
    delta_V.append(np.linalg.norm(total_delta_v))
    delta_Vx.append(total_delta_v[0])
    delta_Vy.append(total_delta_v[1])
    delta_Vz.append(total_delta_v[2])
    print(f'total_delta_v: {total_delta_v}, thrust_body: {thrust_body}')
    return delV, total_delta_v

def compute_velocity_pointing_error(velocity, desired_direction):
    v_norm = np.linalg.norm(velocity)
    d_norm = np.linalg.norm(desired_direction)
    if v_norm == 0 or d_norm == 0:
      return 0.0  # Or np.nan, if undefined
    cos_angle = np.dot(velocity, desired_direction) / (v_norm * d_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # prevent numerical errors
    angle_rad = np.arccos(cos_angle)
    return angle_rad

'''def quat_to_euler(q):
    R = quat_to_dcm(q)
    # 3-1-2 (Z-X-Y) Euler angles
        # Extract Euler angles from DCM: 3-1-2 (Z-X-Y)
    phiz  = np.arctan2(R[1,2], R[2,2])      # Z
    phix  = np.arcsin(-R[0,2])              # X
    phiy  = np.arctan2(R[0,1], R[0,0])      # Y
    return phix, phiy, phiz
def quat_to_dcm(q):
    q0, q1, q2, q3 = q
    return np.array([
         [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),       2*(q1*q3 + q0*q2)],
         [2*(q1*q2 + q0*q3),         1 - 2*(q1**2 + q3**2),   2*(q2*q3 - q0*q1)],
         [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),       1 - 2*(q1**2 + q2**2)]
     ])'''

#Compute attitude errors from the GMAT ephemeris
def compute_attitude_error(q_des, q_est):
    """Compute error quaternion and axis-angle."""
    if np.dot(q_des, q_est) < 0:
        q_est = -q_est  # Flip sign
    q_est_conj = np.array([q_est[0], -q_est[1], -q_est[2], -q_est[3]])
    q_err = quat_multiply(q_des, q_est_conj)
    # Normalize the error quaternion
    q_err = q_err / np.linalg.norm(q_err)
    q0 = np.clip(q_err[0], -1.0, 1.0)
    # Compute angle
    angle_rad = 2 * np.arccos(q0)
    angle_deg = np.degrees(angle_rad)
    # Compute axis
    sin_half_angle = np.sqrt(1 - q0**2)
    if sin_half_angle < 1e-6:
        axis = np.array([0.0, 0.0, 0.0])  # No meaningful rotation axis
    else:
        axis = q_err[1:] / sin_half_angle
    return axis, angle_rad, angle_deg


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions."""
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def pd_control(t, axis, angle, omega, kp, kd):
    """Compute required torque from attitude error and angular velocity."""
    #τ=I⋅α   - maximum torque req based on MOI, time interval and theta.
    #α = 2*theta/(dt^2)  -  deadband is the nominal angular deviation will be set based on the torque range of RCS

    if angle < DEADBAND:
        torque_cmd = - kd * omega
    else:
        if t > control_config['burn_start']+burn_time:
           torque_cmd = -kp * (angle * axis) + (kd * omega)
        else:
           torque_cmd = -kp * (angle * axis) - (kd * omega)
    return torque_cmd

def map_torque_to_thrusters(torques, torque_cmd, I):
    """Optimal Jet Selection: minimize fuel cost while achieving torque_cmd."""
    torque_tolerance = np.linalg.norm(I * DEADBAND)
    best_u = np.zeros(len(thruster_config))
    if np.linalg.norm(torque_cmd) < torque_tolerance:
      return np.array([0, 0]), best_u, 0
    else:
      # Solve binary integer least squares: minimize ||torques @ u - torque_cmd||
      N = torques.shape[0]
      best_thrusters = None
      best_error = np.inf
      best_u = None
      for n_thrusters in [2, 3]:
        for thruster_set in combinations(range(N), n_thrusters):
          A = torques[list(thruster_set)].T
          u, _, _, _ = np.linalg.lstsq(A, torque_cmd, rcond=None)
          if np.all(u >= 0) and np.all(u <= 1):  # Only accept positive duty cycles
              achieved = A @ u
              error = np.linalg.norm(torque_cmd - achieved)
              if error < best_error:
                  best_error = error
                  best_thrusters = thruster_set
                  best_u = u
                  achieved_torque = achieved
    pulse = np.sum(best_u)
    #print(f'torque_tolerance: {torque_tolerance}, best_u: {best_u}, acheived_torque: {achieved_torque}, torque_cmd: {torque_cmd}, pulses : {pulse}')
    return best_thrusters, best_u, pulse

# Van de Har dynamics with spin around Z-axis
def otv_dynamics(t, state, omega_des):
    # --- Unpack State ---

    wx, wy, wz = state[0:3]
    q_est = state[3:7]
    m = state[7]
    h = state[8]
    #Momentx, Momenty, Momentz = state[9:12]
    alpha = state[9]
    delVx, delVy, delVz = state[10:13]

    delV = np.array([delVx, delVy, delVz])
    omega = np.array([wx, wy, wz])
    #omega = np.clip(omega, -0.5, 0.5)
    q_des = Qdes_interp(t)
    #omega_des = AngVels_interp(t)
    q_des = np.array(q_des)
    omega_des = np.array(omega_des)

    omega_error = omega_des - omega

    # --- Mass & Inertia Update ---
    mass_fraction = (m - m_dry) / (initialMass - m_dry)
    I = I_dry + mass_fraction * (I_full - I_dry)

    dalpha = 0
    F = 0

    # --- Phase Determination --- &  --- Main Engine Thrust Profile ---
    if control_config['burn_start'] <= t < control_config['burn_start'] + burn_time:
        phase = "BURN"
        # burnduration Mx (moment from thrust misalignment and motor offset)
        sigmoid = 1 / (1 + np.exp(- sigmoidSlope * (t - sigmoidRiseTime)))
        f_s = sigmoidPostValue * sigmoid + sigmoidPreValue * (1 - sigmoid)
        dalpha = sigmoidFactor * sigmoidSlope * f_s * (1 - f_s)
        t_burn = t - control_config['burn_start']

        # Ramp up/down thrust
        if t_burn <= rampup:
             F = F_max * (t_burn / rampup)
        elif t_burn >= (burn_time - rampup):
             F = F_max * (1 - (t_burn - (burn_time - rampup)) / rampup)
        else:
             F = F_max
        dh = (hf - initialDistanceNozzleThroatCG) / burn_time
        h = initialDistanceNozzleThroatCG + (dh * t_burn)
        Mx = (motorOffset*F*np.cos(alpha)) - (h*F*np.sin(alpha))
        My = F*np.sin(alpha)*h
        Mz = -F*np.cos(alpha)*h
    else:
        phase = "No burn"
        dh = 0
        Mx = 0
        My = 0
        Mz = 0
    #print(Mx,  My)
    Mx = np.clip(Mx, -(THRUST_FORCE*h), (THRUST_FORCE*h))
    q = q_est/np.linalg.norm(q_est)
    q_des = q_des/np.linalg.norm(q_des)
    axis, angle_rad, angle_deg = compute_attitude_error(q_des, q)
    dq = 0.5 * quat_multiply(q, [0, *omega])
    #omega = np.clip(omega, -0.5, 0.5)
    q_new = q + dq * FIRE_CYCLE_TIME
    q_new /= np.linalg.norm(q_new)
    theta_x = axis[0]*angle_deg
    theta_y = axis[1]*angle_deg
    theta_z = axis[2]*angle_deg
    MOI = np.array([I[0,0], I[1,1], I[2,2]])
    if t <  control_config['burn_start']:
       ζ = 0.6
       ω_n = np.array([0.00055, 0.00055, 0.0008])
       #I_now = np.array([I_full[0,0], I_full[1,1], I_full[2,2]])
    elif t > control_config['burn_start']+burn_time:
       ζ = 0.95
       ω_n = np.array([0.021, 0.021, 0.054])
       #I_now = np.array([I_dry[0,0], I_dry[1,1], I_dry[2,2]])
    else:
       # Linear interpolation factor
       inter = (t - control_config['burn_start']) / (control_config['burn_start']+burn_time - control_config['burn_start'])
       # Interpolate ω_n and ζ
       ζ = (1 - inter) * 0.6 + inter * 0.95
       ω_n = (1 - inter) * np.array([0.00055, 0.00055, 0.0008]) + inter * np.array([0.021, 0.021, 0.054])
    # Gain calculation (applies in all branches)
    kp = MOI * ω_n**2
    #kd = 2*ζ*np.sqrt(kp*MOI)
    kd = 2 * ζ * ω_n * MOI
    if np.linalg.norm(omega_error) < 3e-3:
       omega_error = np.array([0, 0, 0])
    torque_rcs = pd_control(t, axis, angle_rad, omega_error, kp, kd) - np.array([Mx, My, Mz]) # = control_config['kp'], kd = control_config['kd'])
    torque_rcs = np.clip(torque_rcs, -(THRUST_FORCE*h), (THRUST_FORCE*h))
    if np.linalg.norm(torque_rcs)< 3e-3 or np.linalg.norm(omega)<3e-3:
       torque_rcs = np.array([0, 0, 0])
    else:
       torque =  np.abs(torque_rcs) < 3e-3
       for i in range (3):
        if torque[i]:
         torque_rcs[i] = 0
    torquex.append(torque_rcs[0])
    torquey.append(torque_rcs[1])
    torquez.append(torque_rcs[2])
    # --- Map torque command to thruster firings ---
    best_thrusters, best_u, pulse = map_torque_to_thrusters(torques, torque_rcs, I)

    # --- Propellant Use: RCS ---
    if pulse is not None:
       rcs_thrust = THRUST_FORCE * pulse
    else:
       rcs_thrust = 0
    rcs.append(rcs_thrust)
    rcs_x.append(torque_rcs[0]/h)
    rcs_y.append(torque_rcs[1]/h)
    rcs_z.append(torque_rcs[2]/h)
    PropMass_rcs = -rcs_thrust/ (CONST_GRAVITATIONAL_ACC * 230)
    m_rcs.append(-PropMass_rcs)
    # --- Total Control Torque ---
    Mx += torque_rcs[0] #+ Tx_gimbal
    My = torque_rcs[1]+np.clip((F*np.sin(alpha)*h), -(THRUST_FORCE*h), (THRUST_FORCE*h)) #+ Ty_gimbal
    Mz = torque_rcs[2]-np.clip((F*np.cos(alpha)*h), -(THRUST_FORCE*h), (THRUST_FORCE*h))
    Moments = np.array([Mx, My, Mz])
    # Moment derivatives assumed zero or modeled externally
    #Momentdot = np.zeros(3)

    # --- Mass Flow Rate ---
    mdot_main = -F / (CONST_GRAVITATIONAL_ACC * I_sp) if F > 0 else 0
    dm = mdot_main + PropMass_rcs
    dwx = Mx / I[0,0] - ((I[2,2] - I[1,1]) / I[0,0]) * wy * wz + (dm * h**2 * wx / I[0,0])
    dwy = My / I[1,1] - ((I[0,0] - I[2,2]) / I[1,1]) * wz * wx + (dm * h**2 * wy / I[1,1])
    dwz = Mz / I[2,2] - ((I[1,1] - I[0,0]) / I[2,2]) * wx * wy

    domega = np.array([dwx, dwy, dwz])

    # --- Quaternion Update ---
    wx_l.append(wx)
    wy_l.append(wy)
    wz_l.append(wz)
    q1 = np.array([q[1], q[2], q[3], q[0]])
    R = Rt.from_quat(q1)
    dcm = R.as_matrix()


    phiz  = np.clip(np.arctan2(dcm[1,2], dcm[2,2]), -1, 1)      # Z
    phix  = np.clip(np.arcsin(-dcm[0,2]), -1, 1)              # X
    phiy  = np.clip(np.arctan2(dcm[0,1], dcm[0,0]), -1, 1)      # Y

    phi = np.array([phix, phiy, phiz])
    phix_l.append(phix)
    phiy_l.append(phiy)
    phiz_l.append(phiz)

    time.append(t)
    Thetax.append(theta_x)
    Thetay.append(theta_y)
    Thetaz.append(theta_z)
    # Velocity dynamics from thrust vector in body frame transformed to inertial
    if control_config['burn_start'] <= t < control_config['burn_start'] + burn_time:
       thrust_body = np.array([0, F*np.sin(alpha)+torque_rcs[1], F*np.cos(alpha)+torque_rcs[2]])
    else:
       thrust_body = np.array([0, F*np.sin(alpha), F*np.cos(alpha)])
    # Rotation matrix from body to inertial using quaternions
    thrust_inertial = dcm.T @ thrust_body

    dv, total_dv = step_callback(t, dcm, thrust_body, m)

    dv_norm = np.linalg.norm(dv)
    efficiency = dv[2]/dv_norm if dv_norm !=0 else 0
    eff.append(efficiency)

    thrust_dir_inertial = thrust_inertial / np.linalg.norm(thrust_inertial) if np.linalg.norm(thrust_inertial) != 0 else 0
    error_rad = compute_velocity_pointing_error(total_dv, thrust_dir_inertial)
    VPE_deg = np.degrees(error_rad)
    VPE.append(VPE_deg)

    V_x.append(dv[0])
    V_y.append(dv[1])
    V_z.append(dv[2])
    mdot.append(dm)
    state_dot = np.zeros_like(state)
    state_dot[0:3] = domega     # Angular acceleration
    state_dot[3:7] = dq        # Quaternion derivative
    state_dot[7] = dm          # Total mass flow
    state_dot[8] = dh
    #state_dot[9:12] = Momentdot
    state_dot[9] = dalpha
    state_dot[10:13] = dv
    print(f't: {t}, omega: {omega}, omega_des: {omega_des}, domega: {domega}')
    print(f'kp: {kp}, kd: {kd}, axis: {axis}, angle: {angle_deg}')
    print(f'torque_rcs: {torque_rcs},Moments: {Moments},m:{m}, VPE: {VPE_deg}, alpha:{alpha}, F:{F} \n\n')
    return state_dot

if __name__ == "__main__":

	t_eval= np.arange(4090, t_uniform[-1]+0.01, 0.01)
	Qdes_interp = interp1d(t_uniform, np.array(Qdes).T, kind='linear', fill_value="extrapolate")
	AngVels_interp = np.vstack([wx_interp(t_eval), wy_interp(t_eval), wz_interp(t_eval)]).T
	Next = 100
	lastPrint = 0
	delVxin = 0
	delVyin = 0
	delVzin = 0
	dt = 0.01

	initial_conditions = [
    initialAngularVelocityX,
    initialAngularVelocityY,
    AngVels_interp[0,2],
    q_esti0, q_esti1, q_esti2, q_esti3,
    initialMass,
    initialDistanceNozzleThroatCG,
    initialThrustMisalignment, delVxin, delVyin, delVzin]

	stateout = np.zeros((len(t_eval),len(initial_conditions)))
	for i in range(len(t_eval)):
	   # Save the current state
	   t_now = t_eval[i]
	   stateout[i,:] = initial_conditions
	   if t_now > lastPrint:
	     lastPrint += Next
	   omega_des = AngVels_interp[i]
	   # 4th Order Runge-Kutta Integrator
	   k1 = otv_dynamics(t_now,initial_conditions, omega_des)
	   k2 = otv_dynamics(t_now+dt/2,initial_conditions+k1*dt/2, omega_des)
	   k3 = otv_dynamics(t_now+dt/2,initial_conditions+k2*dt/2, omega_des)
	   k4 = otv_dynamics(t_now+dt,initial_conditions+k3*dt, omega_des)
	   k = (1.0/6.0)*(k1+2*k2+2*k3+k4)
	   initial_conditions = initial_conditions + k*dt
	mass_dot = list(int(val) for val in  stateout[:,7])
 	# Compare with nominal Delta-V (no misalignment, no spin)
	idealDeltaV = []
	for i in range (len(t_eval)):
	   if t_eval[i] > control_config['burn_start'] and t_eval[i] < control_config['burn_start']+burn_time:
	     if mass_dot[i] > 0:
	      dV =  (CONST_GRAVITATIONAL_ACC * I_sp) * np.log(initialMass/mass_dot[i])
	     else:
	      dV = 0
	   else:
	     dV = 0
	   idealDeltaV.append(dV)
	# Error
	deltaV_error = idealDeltaV[-1] - delta_V[-1]

	df = pd.DataFrame({'BurnTime':time, 'Thetax':Thetax, 'Thetay':Thetay, 'Thetaz': Thetaz, 'Torquex':torquex, 'Torquey':torquey, 'Torquez':torquez})

	plt.figure(figsize=(8, 6))
	thetax_smooth = gaussian_filter1d(Thetax, sigma=3)
	thetay_smooth = gaussian_filter1d(Thetay, sigma=3)
	thetaz_smooth = gaussian_filter1d(Thetaz, sigma=3)
	plt.plot(time[::1000], thetax_smooth[::1000], label='Theta_x Deg')
	plt.plot(time[::1000], thetay_smooth[::1000], label='Theta_y Deg')
	plt.plot(time[::1000], thetaz_smooth[::1000], label='Theta_z Deg')
	plt.xlabel('Time [s]')
	plt.title('RCS angle [Deg] Over Time')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	Torqx_smooth = gaussian_filter1d(torquex, sigma=3)
	Torqy_smooth = gaussian_filter1d(torquey, sigma=3)
	Torqz_smooth = gaussian_filter1d(torquez, sigma=3)
	plt.plot(time[::1000], Torqx_smooth[::1000], label='Torque_x Nm', linestyle='--')
	plt.plot(time[::1000], Torqy_smooth[::1000], label='Torque_y Nm', linestyle='--')
	plt.plot(time[::1000], Torqz_smooth[::1000], label='Torque_z Nm', linestyle='--')
	plt.xlabel('Time [s]')
	plt.title('RCS Control Torque [Nm] Over Time')
	plt.legend()
	plt.grid()


	plt.figure(figsize=(8, 6))
	wx_smooth = gaussian_filter1d(wx_l,  sigma=3)
	wy_smooth = gaussian_filter1d(wy_l,  sigma=3)
	wz_smooth = gaussian_filter1d(wz_l,  sigma=3)
	plt.plot(time[::1000], wx_smooth[::1000], label='w_x rad/sec')
	plt.plot(time[::1000], wy_smooth[::1000], label='w_y rad/sec')
	plt.plot(time[::1000], wz_smooth[::1000], label='w_z rad/sec')
	plt.xlabel('Time [s]')
	plt.title('Angular velocity (rad/sec) RCS Control')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	'''phix_smooth = gaussian_filter1d(phix_l,  sigma=3)
	phiy_smooth = gaussian_filter1d(phiy_l,  sigma=3)
	phiz_smooth = gaussian_filter1d(phiz_l,  sigma=3)
	plt.plot(time[::1000], phix_smooth[::1000], label='phi_x rad')
	plt.plot(time[::1000], phiy_smooth[::1000], label='phi_y rad')'''
	plt.plot(time[::1000], phiz_l[::1000], label='phi_z rad')
	plt.xlabel('Time [s]')
	plt.title('Euler angle ψ (rad)')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	plt.plot(time[::1000], delta_Vx[::1000], label='deltaVx (Lateral-roll)')
	plt.plot(time[::1000], delta_Vy[::1000], label='deltaVy (Normal-pitch)')
	plt.plot(time[::1000], delta_Vz[::1000], label='deltaVz (spin&Thrust)')
	plt.plot(time[::1000], delta_V[::1000], label='|DeltaV| magnitude', linestyle='--')
	plt.plot(t_eval, idealDeltaV, label='|idealDeltaV|', linestyle='--')
	plt.xlabel('Time [s]')
	plt.ylabel('Velocity [m/s]')
	plt.title('Velocity Components_RCS Control')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	rcs_smooth = gaussian_filter1d(rcs,  sigma=3)
	plt.plot(time[::1000], rcs_smooth[::1000], label='rcs Thrust in N')
	plt.xlabel('Time [s]')
	plt.title('Fuel optimized RCS Thrust profile')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	rcs_x_smooth = gaussian_filter1d(rcs_x,  sigma=3)
	rcs_y_smooth = gaussian_filter1d(rcs_y,  sigma=3)
	rcs_z_smooth = gaussian_filter1d(rcs_z,  sigma=3)
	plt.plot(time[::1000], rcs_x_smooth[::1000], label='RCS Thrust X axis')
	plt.plot(time[::1000], rcs_y_smooth[::1000], label='RCS Thrust Y axis')
	plt.plot(time[::1000], rcs_z_smooth[::1000], label='RCS Thrust Z axis')
	plt.xlabel('Time [s]')
	plt.title('RCS 3-Axis Thrust profile in N')
	plt.grid()
	plt.tight_layout()
	plt.show()

	plt.figure(figsize=(8, 6))
	VPE_smooth = gaussian_filter1d(VPE,  sigma=3)
	plt.plot(time[::1000], VPE_smooth[::1000], label='Velocity Pointing error in Deg')
	plt.xlabel('Time [s]')
	plt.title('Velocity pointing Error_RCS Control')
	plt.grid()

	plt.figure(figsize=(8, 6))
	mrcs_smooth = gaussian_filter1d(m_rcs,  sigma=3)
	plt.plot(time[::500], mrcs_smooth[::500], label='rcs massflow rate used in Kg/s')
	plt.xlabel('Time [s]')
	plt.title('Fuel optimized mass flow rate_RCS control')
	plt.legend()
	plt.grid()
	plt.tight_layout()
	plt.show()
