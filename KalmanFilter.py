"""
EKF inside ODE solver demo for spacecraft with RCS pulses + Reaction Wheels.

- Nominal dynamics integrated with scipy.solve_ivp (true state and estimator nominal).
- Error-state multiplicative EKF (3 attitude error, 3 rate error, 3 wheel-speed error).
- Measurements: quaternion (star tracker) + body rates (gyro) + wheel speeds.
- Simple RCS torque pulses used as example external torques.
- Simple RW torque command used for wheel dynamics coupling.

Dependencies:
    pip install numpy scipy matplotlib

Run:
    python ekf_rcs_rw_demo.py
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------- Utility functions -------------------------
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def quat_mul(q1, q2):
    # q = [q0, qv] where q0 scalar
    w1 = q1[0]; v1 = q1[1:]
    w2 = q2[0]; v2 = q2[1:]
    w = w1*w2 - np.dot(v1, v2)
    v = w1*v2 + w2*v1 + np.cross(v1, v2)
    return np.hstack((w, v))

def quat_conjugate(q):
    return np.hstack((q[0], -q[1:]))

def quat_to_rotvec(q):
    qn = q / np.linalg.norm(q)
    w = qn[0]; v = qn[1:]
    angle = 2*np.arccos(np.clip(w, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = v / np.sin(angle/2.0)
    return axis * angle

def normalize_quat(q):
    return q / np.linalg.norm(q)

# ------------------------- Dynamics definitions -------------------------
# Example inertia (kg·m^2) - replace with your vehicle inertia
I = np.diag([15.4667, 15.4667, 4.2667])
I_inv = np.linalg.inv(I)

# Reaction wheel inertias (single-axis wheels aligned with body axes)
I_rw = np.array([0.1, 0.1, 0.05])  # kg·m^2

def attitude_kinematics(q, omega):
    # dq/dt = 0.5 * Omega(omega) * q
    q0 = q[0]; qv = q[1:]
    Omega = np.zeros((4,4))
    Omega[0,1:] = -omega
    Omega[1:,0] =  omega
    Omega[1:,1:] = -skew(omega)
    return 0.5 * Omega.dot(q)

def spacecraft_dynamics(t, state, control_torque_func):
    """
    state: [q0,q1,q2,q3, wx,wy,wz, wr1,wr2,wr3]
    control_torque_func: function(t) -> 3-vector of body torques (RCS)
    """
    q = state[0:4]
    w = state[4:7]
    wr = state[7:10]

    # External RCS torque (body frame)
    tau_rcs = control_torque_func(t)

    # Reaction wheel torques (torque applied to each wheel; body feels equal and opposite)
    tau_rw = rw_torque_command(t, w, wr)  # torque applied to each wheel (positive increases wr)
    total_rw_torque_body = np.array([-tau_rw[0], -tau_rw[1], -tau_rw[2]])

    # rotational dynamics: I * wdot = - w x (I w) + tau_rcs + total_rw_torque_body
    wdot = I_inv.dot(-np.cross(w, I.dot(w)) + tau_rcs + total_rw_torque_body)

    # wheel dynamics: I_rw * dwr/dt = tau_rw
    dwr = tau_rw / I_rw

    dq = attitude_kinematics(q, w)
    dstatedt = np.hstack((dq, wdot, dwr))
    return dstatedt

# Example RW torque command (simple damping)
def rw_torque_command(t, w, wr):
    # Damping-like torque: torque on wheel = -Kd * body_rate
    Kd = np.array([0.05, 0.05, 0.02])  # tune to your system
    tau = -Kd * w
    return tau

# Example RCS torques (pulses)
def rcs_control_torque(t):
    # Example pulses at t ~ 5s and 20s
    if 5.0 < t < 5.05:
        return np.array([0.0, 0.0, 2.0])
    if 20.0 < t < 20.02:
        return np.array([0.5, -0.8, 0.0])
    return np.zeros(3)

# ------------------------- Simulation & Measurements -------------------------
t0 = 0.0
tf = 60.0

# Initial true state
q0_true = normalize_quat(np.array([1.0, 0.0, 0.0, 0.0]))
w0_true = np.deg2rad(np.array([0.2, -0.3, 0.1]))  # rad/s
wr0_true = np.array([100.0, -80.0, 50.0])  # wheel speeds (rad/s)
state0 = np.hstack((q0_true, w0_true, wr0_true))

# Measurement times and rates
meas_dt = 0.5  # measurement cadence (s)
meas_times = np.arange(t0+meas_dt, tf+1e-6, meas_dt)

# Measurement noise characteristics
sigma_star_tracker = 1e-3  # small-angle equivalent (rad)
sigma_gyro = np.deg2rad(0.05)  # rad/s gyro noise std
sigma_wheel = 0.01  # wheel-speed measurement noise

def generate_measurement(true_state):
    q = true_state[0:4]; w = true_state[4:7]; wr = true_state[7:10]
    # quaternion measurement with small rotation noise
    rotvec_noise = np.random.normal(scale=sigma_star_tracker, size=3)
    angle = np.linalg.norm(rotvec_noise)
    if angle < 1e-12:
        q_noise = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = rotvec_noise / angle
        q_noise = np.hstack((np.cos(angle/2.0), axis*np.sin(angle/2.0)))
    q_meas = normalize_quat(quat_mul(q_noise, q))
    # gyro measurement (body rate + noise)
    w_meas = w + np.random.normal(scale=sigma_gyro, size=3)
    # wheel speed measurement
    wr_meas = wr + np.random.normal(scale=sigma_wheel, size=3)
    return np.hstack((q_meas, w_meas, wr_meas))

# ------------------------- EKF (error-state multiplicative) -------------------------
# Error-state: delta_theta (3), delta_w (3), delta_wr (3) => 9x1

# Estimator initial nominal state (intentionally mismatched)
q_hat = normalize_quat(np.array([1.0, 0.0, 0.0, 0.0]))
w_hat = np.zeros(3)
wr_hat = np.zeros(3)

# Estimator covariance
P = np.eye(9) * 1e-3

# Continuous process noise for error-state
Q_cont = np.diag(np.hstack((np.ones(3)*1e-6, np.ones(3)*1e-6, np.ones(3)*1e-4)))

# Measurement covariance
R_meas = np.diag(np.hstack((np.ones(3)*(sigma_star_tracker**2),
                            np.ones(3)*(sigma_gyro**2),
                            np.ones(3)*(sigma_wheel**2))))

def propagate_nominal(t0, tf, q_init, w_init, wr_init):
    def dyn(t, s):
        return spacecraft_dynamics(t, s, rcs_control_torque)
    s0 = np.hstack((q_init, w_init, wr_init))
    sol = solve_ivp(dyn, (t0, tf), s0, rtol=1e-8, atol=1e-10, max_step=0.02)
    s_final = sol.y[:, -1]
    s_final[0:4] = normalize_quat(s_final[0:4])
    return s_final

def compute_A_matrix(q, w, wr, tau_rw):
    A = np.zeros((9,9))
    # delta_theta_dot ≈ -skew(w) * delta_theta + delta_w
    A[0:3,0:3] = -skew(w)
    A[0:3,3:6] = np.eye(3)
    # delta_w_dot approx = -I_inv * skew(I*w) * delta_w
    A[3:6,3:6] = -I_inv.dot(skew(I.dot(w)))
    # wheels block small/neglected for simplicity
    A[6:9,6:9] = np.zeros((3,3))
    return A

def discretize_AQ(A, Qc, dt):
    Ad = np.eye(A.shape[0]) + A*dt
    Qd = Qc*dt
    return Ad, Qd

def measurement_update(q_hat, w_hat, wr_hat, P, z_meas):
    q_meas = z_meas[0:4]
    w_meas = z_meas[4:7]
    wr_meas = z_meas[7:10]

    q_err = quat_mul(q_meas, quat_conjugate(q_hat))
    delta_theta_meas = 2.0 * q_err[1:]  # small-angle approx
    y = np.hstack((delta_theta_meas, w_meas - w_hat, wr_meas - wr_hat))

    H = np.zeros((9,9))
    H[0:3,0:3] = np.eye(3)
    H[3:6,3:6] = np.eye(3)
    H[6:9,6:9] = np.eye(3)

    S = H.dot(P).dot(H.T) + R_meas
    K = P.dot(H.T).dot(np.linalg.inv(S))
    delta_x = K.dot(y)
    P_upd = (np.eye(9) - K.dot(H)).dot(P)

    delta_theta = delta_x[0:3]
    delta_w = delta_x[3:6]
    delta_wr = delta_x[6:9]

    # small-angle to quaternion
    angle = np.linalg.norm(delta_theta)
    if angle < 1e-12:
        dq = np.hstack((1.0, 0.5*delta_theta))
    else:
        axis = delta_theta / angle
        dq = np.hstack((np.cos(angle/2.0), axis*np.sin(angle/2.0)))

    q_hat_new = normalize_quat(quat_mul(dq, q_hat))
    w_hat_new = w_hat + delta_w
    wr_hat_new = wr_hat + delta_wr

    return q_hat_new, w_hat_new, wr_hat_new, P_upd

# ------------------------- Run simulation with EKF -------------------------
np.random.seed(1)

t = t0
true_state = state0.copy()

# storage for plotting
times = [t0]
true_ws = [true_state[4:7].copy()]
est_ws = [w_hat.copy()]
att_err_deg = [np.rad2deg(np.linalg.norm(quat_to_rotvec(quat_mul(true_state[0:4], quat_conjugate(q_hat)))))]
q_history_true = [true_state[0:4].copy()]
q_history_hat = [q_hat.copy()]

for k in range(len(meas_times)):
    t_next = meas_times[k]
    # propagate true state
    sol = solve_ivp(lambda tt, s: spacecraft_dynamics(tt, s, rcs_control_torque),
                    (t, t_next), true_state, rtol=1e-8, atol=1e-10, max_step=0.02)
    true_state = sol.y[:, -1]
    true_state[0:4] = normalize_quat(true_state[0:4])

    # propagate nominal estimator state (same dynamics)
    s_nom_final = propagate_nominal(t, t_next, q_hat, w_hat, wr_hat)
    q_hat = s_nom_final[0:4]
    w_hat = s_nom_final[4:7]
    wr_hat = s_nom_final[7:10]

    # propagate covariance: linearize at nominal and discretize
    A = compute_A_matrix(q_hat, w_hat, wr_hat, None)
    dt = t_next - t
    Ad, Qd = discretize_AQ(A, Q_cont, dt)
    P = Ad.dot(P).dot(Ad.T) + Qd

    # measurement and update
    z = generate_measurement(true_state)
    q_hat, w_hat, wr_hat, P = measurement_update(q_hat, w_hat, wr_hat, P, z)

    # record
    t = t_next
    times.append(t)
    true_ws.append(true_state[4:7].copy())
    est_ws.append(w_hat.copy())
    att_err_deg.append(np.rad2deg(np.linalg.norm(quat_to_rotvec(quat_mul(true_state[0:4], quat_conjugate(q_hat))))))
    q_history_true.append(true_state[0:4].copy())
    q_history_hat.append(q_hat.copy())

# convert to arrays
times = np.array(times)
true_ws = np.array(true_ws)
est_ws = np.array(est_ws)
att_err_deg = np.array(att_err_deg)

# ------------------------- Plots -------------------------
plt.figure(figsize=(10,6))
plt.plot(times, true_ws[:,0], label='true wx')
plt.plot(times, est_ws[:,0], label='est wx', linestyle='--')
plt.plot(times, true_ws[:,1], label='true wy')
plt.plot(times, est_ws[:,1], label='est wy', linestyle='--')
plt.plot(times, true_ws[:,2], label='true wz')
plt.plot(times, est_ws[:,2], label='est wz', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Body rates [rad/s]')
plt.title('True vs Estimated Body Rates (EKF)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(times, att_err_deg)
plt.xlabel('Time [s]')
plt.ylabel('Attitude error [deg]')
plt.title('Attitude estimation error (angle)')
plt.grid(True)
plt.show()

print('Final attitude error (deg):', att_err_deg[-1])
print('Final rate error (rad/s):', np.linalg.norm(true_ws[-1]-est_ws[-1]))
