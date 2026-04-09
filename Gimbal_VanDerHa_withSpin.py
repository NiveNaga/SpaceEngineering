import numpy as np
import pandas as pd
import os, sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from types import SimpleNamespace
# Constants
CONST_GRAVITATIONAL_ACC = 9.8065  # m/s^2
#specificImpulse = 310  # s
specificImpulse = 280  # s

# Initial parameters
initialMass = 571  # kg
initialMomentsOfInertiaX = 266.4938    #25.1  # kg*m^2
initialMomentsOfInertiaY = 266.4938  #35.4
initialMomentsOfInertiaZ = 190.3429  #38.2
initialAngularVelocityX = 0.00  # rad/s
initialAngularVelocityY = 0.00  # rad/s (spin applied here)
initialAngularVelocityZ = 3.14  # rad/s
#initialAngularVelocityZ = 6.28  # rad/s

# Euler angles (3-1-2 sequence)
initialEulerAnglePhiX = 0  # rad
initialEulerAnglePhiY = 0
initialEulerAnglePhiZ = 0

# Inertial velocities [m/s]
initialInertialVelocityX = 0
initialInertialVelocityY = 0
initialInertialVelocityZ = 0

# Thrust and misalignment

maxThrust = 2153  # N
initialDistanceNozzleThroatCG = 0.5
motorOffset = 0.01
initialThrustMisalignment = np.deg2rad(0)  # rad
# Burn duration
burnTime = 300  # seconds
momentY = 0
momentZ = 0
rampup = 10
initialThrust = 0

#FinalParameters
finalDistanceNozzleThroatCG = 0.7
#Thrust misalignment: Parameter setting for sigmoid functions
sigmoidRiseTime = 50 # Time to start a stepwise change in the sigmoid function
sigmoidPreValue = 0 # Value before the step
sigmoidPostValue= 1 # Value after the step
sigmoidSlope = 10 # Slope of the sigmoid function
thrustMisalignmentFactor = 2
#thrustMisalignmentFactor = 0.02
sigmoidFactor = thrustMisalignmentFactor * np.pi / 180

kRatioX = initialMomentsOfInertiaX / initialMass
kRatioY = initialMomentsOfInertiaY / initialMass
kRatioZ = initialMomentsOfInertiaZ / initialMass

Thetax, Thetay, torquex, torquey, time, wx_l, wy_l, wz_l, phix_l, phiy_l, phiz_l, VPE, delta_V, delta_Vx, delta_Vy, delta_Vz = [[] for _ in range(16)]

def compute_velocity_pointing_error(velocity, desired_direction):
     v_norm = np.linalg.norm(velocity)
     d_norm = np.linalg.norm(desired_direction)
     if v_norm == 0 or d_norm == 0:
       return 0.0  # Or np.nan, if undefined
     cos_angle = np.dot(velocity, desired_direction) / (v_norm * d_norm)
     cos_angle = np.clip(cos_angle, -1.0, 1.0)  # prevent numerical errors
     angle_rad = np.arccos(cos_angle)
     return angle_rad

total_delta_v = np.array([0.0, 0.0, 0.0])
last_time = 0
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
     return delV, total_delta_v

# Van de Har dynamics with spin around Y-axis
def van_de_har_dynamics(t, y):
    h, alpha, Mx, m, Ix, Iy, Iz, wx, wy, wz, phix, phiy, phiz, Vx, Vy, Vz = y

    # Assume constant thrust for simplicity
    if t <= rampup:
         F = maxThrust*(t/rampup)
         '''elif t >= (burnTime-rampup):
         F = maxThrust*(1-(t-(burnTime-rampup))/rampup)'''
    else:
         F = maxThrust
    dh = (finalDistanceNozzleThroatCG - initialDistanceNozzleThroatCG) / burnTime
    sigmoid = 1 / (1 + np.exp(- sigmoidSlope * (t - sigmoidRiseTime)))
    f_s = sigmoidPostValue * sigmoid + sigmoidPreValue * (1 - sigmoid)
    dalpha = sigmoidFactor * sigmoidSlope * f_s * (1 - f_s)
    dMx = F * ((h * np.cos(alpha) - motorOffset * np.sin(alpha)) * dalpha + dh * np.sin(alpha))
    dm = - F/(CONST_GRAVITATIONAL_ACC * specificImpulse)
    dIx = kRatioX * dm
    dIy = kRatioY * dm
    dIz = kRatioZ * dm

    # Angular velocity dynamics
    dwx1 = Mx / Ix - ((Iz - Iy) / Ix) * wy * wz + (dm * h**2 * wx / Ix)
    dwy1 = momentY / Iy - ((Ix - Iz) / Iy) * wz * wx + (dm * h**2 * wy / Iy)
    dwz1 = momentZ / Iz - ((Iy - Ix) / Iz) * wx * wy
    # Introducing PD control using RW and Gimbal
    #Kp_rw = 4   # proportional gain for RW
    #Kd_rw = 30  # derivative gain for RW
    Kp_g_base = 1  # proportional gain for gimbal
    Kd_g_base = 2.0 # derivative gain for gimbal

    # Gain scheduling for first few seconds
    #if t < 5.0:
       #Kp_g = 2 * Kp_g_base
       #Kd_g = 2 * Kd_g_base
    #else:
    Kp_g = Kp_g_base
    Kd_g = Kd_g_base

    # --- Attitude Errors (desired angles = 0) ---
    error_phi_x = -phix
    error_phi_y = -phiy

    # --- Optional: Deadband to avoid small jitters ---
    deadband_rad = np.radians(0.1)
    if abs(error_phi_x) < deadband_rad: error_phi_x = 0
    if abs(error_phi_y) < deadband_rad: error_phi_y = 0

    # --- Gimbal Commands ---
    thetax_cmd = (Kp_g * error_phi_x - Kd_g * wx)
    thetay_cmd = (Kp_g * error_phi_y - Kd_g * wy)
    # --- Clamp Gimbal Angles ---
    max_gimbal_rad = 10  # limit to ±0.175 rads

    thetax = np.clip(thetax_cmd, -(np.deg2rad(max_gimbal_rad)), np.deg2rad(max_gimbal_rad))
    thetay = np.clip(thetay_cmd, -(np.deg2rad(max_gimbal_rad)), np.deg2rad(max_gimbal_rad))

    Tx_control = F*h*np.sin(thetax)
    Ty_control = F*h*np.sin(thetay)
    Tz_control = 0
    dwx = (Mx + Tx_control) / Ix - ((Iz - Iy) / Ix) * wy * wz + (dm * h**2 * wx / Ix)
    dwy = Ty_control / Iy - ((Ix - Iz) / Iy) * wz * wx + (dm * h**2 * wy / Iy)
    dwz = ((Iy - Ix) / Iz) * wx * wy
    # Euler angle kinematics Control profile plot
    Thetax.append(np.rad2deg(thetax))
    time.append(t)
    Thetay.append(np.rad2deg(thetay))
    torquex.append(Tx_control)
    torquey.append(Ty_control)
    #print(F, h, np.degrees(thetax), np.degrees(thetay), Tx_control, Ty_control)
    dphix = wx * np.cos(phiy) + wz * np.sin(phiy)
    dphiy = wy - (wz * np.cos(phiy) - wx * np.sin(phiy)) * np.tan(phix)
    dphiz = (wz * np.cos(phiy) - wx * np.sin(phiy)) / np.cos(phix)


    wx_l.append(dwx)
    wy_l.append(dwy)
    wz_l.append(dwz)

    phix_l.append(dphix)
    phiy_l.append(dphiy)
    phiz_l.append(dphiz)
    # Velocity dynamics from thrust vector in body frame transformed to inertial
    thrust_body = np.array([0, 0, F])  # Assume thrust in X-body direction

    # Rotation matrix from body to inertial using 3-1-2 Euler angles
    s1, s2, s3 = np.sin(phiz), np.sin(phix), np.sin(phiy)
    c1, c2, c3 = np.cos(phiz), np.cos(phix), np.cos(phiy)

    R = np.array([
        [c3*c1 - s3*s2*s1, s1*c3 + c1*s3*s2, -c2*s3],
        [-c2*s1, c2*c1, s2],
        [s3*c1 + s2*s1*c3, s3*s1-s2*c3*c1, c3*c2]
     ])
    thrust_inertial = R @ thrust_body
    dv, total_dv = step_callback(t, R, thrust_body, m)
    thrust_dir_inertial = thrust_inertial / np.linalg.norm(thrust_inertial) if np.linalg.norm(thrust_inertial) != 0 else 0
    error_rad = compute_velocity_pointing_error(total_dv, thrust_dir_inertial)
    VPE_deg = np.degrees(error_rad)
    VPE.append(VPE_deg)
    dVx, dVy, dVz = thrust_inertial / m
    return [dh, dalpha, dMx, dm, dIx, dIy, dIz,
            dwx, dwy, dwz,
            dphix, dphiy, dphiz,
            dVx, dVy, dVz]

if __name__ == "__main__":
	# Initial Mx (moment from thrust misalignment and motor offset)
	Mx = maxThrust * (initialDistanceNozzleThroatCG * np.sin(initialThrustMisalignment) +
                  motorOffset * np.cos(initialThrustMisalignment))
	initialDeltaV = (CONST_GRAVITATIONAL_ACC * specificImpulse) * np.log(initialMass/initialMass)
	initial_conditions = [
    initialDistanceNozzleThroatCG,
    initialThrustMisalignment,
    Mx,
    initialMass,
    initialMomentsOfInertiaX,
    initialMomentsOfInertiaY,
    initialMomentsOfInertiaZ,
    initialAngularVelocityX,
    initialAngularVelocityY,
    initialAngularVelocityZ,
    initialEulerAnglePhiX,
    initialEulerAnglePhiY,
    initialEulerAnglePhiZ,
    initialInertialVelocityX,
    initialInertialVelocityY,
    initialInertialVelocityZ]

	# Time span
	t_span1 = (0, burnTime)
	t_eval1 = np.linspace(*t_span1, 600)

	# Solve ODE
	sol = solve_ivp(
    fun=van_de_har_dynamics,
    t_span=t_span1,
    y0=initial_conditions,
    method='DOP853',
    t_eval=t_eval1,
    rtol=1e-12,
    atol=1e-12
)
	mass_dot = list(int(val) for val in sol.y[3])
	print(mass_dot[-1])
	# Compare with nominal Delta-V (no misalignment, no spin)
	idealDeltaV = []
	for i in range (0, len(sol.t)):
	    dV =  (CONST_GRAVITATIONAL_ACC * specificImpulse) * np.log(initialMass/mass_dot[i])
	    idealDeltaV.append(dV)
	# Error
	deltaV_error = idealDeltaV[-1] - delta_V[-1]

	df = pd.DataFrame({'BurnTime':time, 'Thetax':Thetax, 'Thetay':Thetay, 'Torquex':torquex, 'Torquey':torquey})
	#df.to_csv(os.getcwd()+'/control.csv', index=False)
	plt.figure(figsize=(8, 6))
	thetax_smooth = gaussian_filter1d(Thetax, sigma=3)
	thetay_smooth = gaussian_filter1d(Thetay, sigma=3)
	plt.plot(time[::35], thetax_smooth[::35], label='Theta_x Deg')
	plt.plot(time[::35], thetay_smooth[::35], label='Theta_y Deg')
	plt.xlabel('Time [s]')
	plt.title('GimbalActuators angle [Deg] Over Time')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	Torqx_smooth = gaussian_filter1d(torquex, sigma=3)
	Torqy_smooth = gaussian_filter1d(torquey, sigma=3)
	plt.plot(time, Torqx_smooth, label='Torque_x Nm', linestyle='--')
	plt.plot(time, Torqy_smooth, label='Torque_y Nm', linestyle='--')
	plt.xlabel('Time [s]')
	plt.title('Control Torque [Nm]_GimbalControl')
	plt.legend()
	plt.grid()


	plt.figure(figsize=(8, 6))
	wx_smooth = gaussian_filter1d(wx_l,  sigma=3)
	wy_smooth = gaussian_filter1d(wy_l,  sigma=3)
	wz_smooth = gaussian_filter1d(wz_l,  sigma=3)
	plt.plot(time, wx_smooth, label='w_x rad/sec')
	plt.plot(time, wy_smooth, label='w_y rad/sec')
	plt.plot(time, wz_smooth, label='w_z rad/sec')
	plt.xlabel('Time [s]')
	plt.title('Angular velocity (rad/sec)_Gimbalcontrol')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	phix_smooth = gaussian_filter1d(phix_l,  sigma=3)
	phiy_smooth = gaussian_filter1d(phiy_l,  sigma=3)
	phiz_smooth = gaussian_filter1d(phiz_l,  sigma=3)
	plt.plot(time, phix_smooth, label='phi_x rad')
	plt.plot(time, phiy_smooth, label='phi_y rad')
	plt.plot(time, phiz_smooth, label='phi_z rad')
	plt.xlabel('Time [s]')
	plt.title('Euler angles (rad)')
	plt.legend()
	plt.grid()

	#print(deltaV_final, idealDeltaV, deltaV_error)
	plt.figure(figsize=(8, 6))
	plt.plot(time, delta_Vx, label='Vx (along thrust)')
	plt.plot(time, delta_Vy, label='Vy (lateral)')
	plt.plot(time, delta_Vz, label='Vz (vertical)')
	plt.plot(time, delta_V, label='|DeltaV| magnitude', linestyle='--')
	plt.plot(sol.t, idealDeltaV, label='|idealDeltaV|', linestyle='--')
	plt.xlabel('Time [s]')
	plt.ylabel('Velocity [m/s]')
	plt.title('Velocity Components (m/s)_GimbalControl')
	plt.legend()
	plt.grid()

	plt.figure(figsize=(8, 6))
	plt.plot(time[::35], VPE[::35], label='Velocity Pointing error in Deg')
	plt.xlabel('Time [s]')
	plt.title('Velocity pointing Error (Deg)_GimbalControl')
	plt.grid()
	plt.tight_layout()
	plt.show()
