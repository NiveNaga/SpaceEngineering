import os, sys
import subprocess
import getopt
import numpy as np
from scipy import integrate

path = os.getcwd()+'/parameters.py'
rho_Al = 2700
y_Al = 3e+8
Sf = 1.5
ullage = 0.2

#st316
rho_st = 7850
y_st = 3e+8
rho_liner = 8900
rho_ins = 200

rho_in718 = 8190
y_in718 = 9e+8

AR = 35
#molecular mass of He in Kg/mol
M_He = 0.004
R = 8.314472
theta = 15

def opts(argv):
	try:
	   opts, args = getopt.getopt(argv, "f:o:p:r:t:m:", ['Fuel=', 'Oxidizer=', 'Pc=', 'Rt=', 'tb=', 'mp='])
	except:
	   pass
	for opt, arg in opts:
	   if opt == '-f':
	     Fuel = arg
	   elif opt == '-o':
	     Oxidizer = arg
	   elif opt == '-p':
	     pc = int(arg)
	   elif opt == '-r':
	     Rt = float(arg)
	   elif opt == '-t':
	     tb = int(arg)
	   elif opt == '-m':
             mp = int(arg)
	return Fuel, Oxidizer, pc, Rt, tb, mp

#PROPELLANT MASS CALCULATION
def prop_Mass(mdot, OF, tb, Ox, Fu):
	mdot_fu = mdot/(OF+1)
	mdot_ox = mdot-mdot_fu
	if Fu == 'C3H6':
	   m_fu = mdot_fu*tb*1.1
	else:
	   m_fu = mdot_fu*tb*1.07
	if (Ox == 'N2O'):
	   m_ox = mdot_ox*tb*1.1
	else:
	   m_ox = mdot_ox*tb*1.07
	print(f'mdot_fu:{mdot_fu}, mdot_ox: {mdot_ox}')
	return mdot_ox, mdot_fu, m_fu, m_ox

#PROPULSION  TANK MASS CALCULATION
def tank_Mass(pc, Rt, tb, mdot_ox, m_fu, m_ox, rho_ox, rho_fu, code, n, a):
	V_oxtank = (m_ox/rho_ox)*1.2
	'''assuming cylindrical tank with L/D = 3'''
	D_oxtank = ((4*V_oxtank)/(3*np.pi))**(1/3)
	L_oxtank = 3*D_oxtank
	#tank thickness using hoops law
#	if Ox != 'N2O':
	t_oxtank = (pc*15*D_oxtank)/(2*y_Al)
#	else:
#	   t_oxtank = (pc*20*D_oxtank*Sf)/(2*y_Al)
	A_oxtank = np.pi*((D_oxtank+(2*t_oxtank))**2-D_oxtank**2)/4
#	A_oxtank = (np.pi*D_oxtank*L_oxtank)+(2*np.pi*(D_oxtank**2/4))
	m_oxtank = A_oxtank*L_oxtank*rho_Al
##	mli_oxtank = 0.5*A_oxtank
##	mo = m_oxtank+mli_oxtank
	if code == 'H':
	   #fuel case material Stainless steel 318
	   Dp = 3*2*Rt
	   Df = ((1+(((2*n+1)*(2**(2*n+1))*a*(mdot_ox**n)*tb)/((Dp**(2*n+1))*(np.pi**n))))**(1/(2*n+1)))*Dp
	   Lf = (4*m_fu)/(np.pi*rho_fu*(Df**2 - Dp**2))
	   print(f'Dp:{Dp}, Df: {Df}, Lf: {Lf}')
	   res_fu = ((Df/2)**2+(0.1*m_fu/(rho_fu*np.pi*Lf)))**(1/2)-(Df/2)
	   V_futank = (np.pi*(Df**2-Dp**2)*Lf)/4
	   tliner = pc*5*Sf*Df/(2*880*(10**6))
	   tins = 0.005
	   t_futank = (pc*5*(Df+(2*tliner)+(2*0.005))*Sf)/(2*y_st)
	   A_liner = np.pi*((Df+(2*tliner))**2-(Df**2))/4
	   A_ins = np.pi*((Df+(2*tliner)+(2*0.005))**2-(Df+(2*tliner))**2)/4
	   A_futank = np.pi*((Df+(2*t_futank)+(2*tliner)+(2*0.005))**2-(Df+(2*tliner)+(2*0.005))**2)/4
#	   A_futank = (np.pi*Df*Lf)+(2*np.pi*(Df**2/4))
	   m_futank = (A_futank*Lf*rho_st)+(A_liner*Lf*4430)+(A_ins*Lf*200)
	   mli_futank = 0
	else:
	   res_fu = 0
	   V_futank = (m_fu/rho_fu)*1.2
	   Df = ((4*V_futank)/(3*np.pi))**(1/3)
	   Lf = 3*Df
#	   if code == 'D':
	   tliner = pc*10*Sf*Df/(2*880*(10**6))
	   tcomp = pc*10*Sf*(Df+(2*tliner))/(2*6800*(10**6))
	   t_futank = tliner+tcomp
#	   t_futank = (pc*15*Df)/(2*y_Al)
#	   else:
#	      t_futank = (pc*15*Df*Sf)/(2*y_Al)
	   Aliner = np.pi*((Df+(2*tliner))**2-Df**2)/4
	   Acomp =  np.pi*((Df+(2*tliner)+(2*tcomp))**2-(Df+(2*tliner))**2)/4
	   mliner = Aliner*Lf*4430
	   mcomp = Lf*1600*Acomp
#	   A_futank = (np.pi*t_futank*Lf)+(2*np.pi*(((Df+t_futank)**2-Df**2)/4))
	   m_futank = mliner+mcomp
	   mli_futank = 0.5*Aliner
	tliner = pc*10*Sf*D_oxtank/(2*880*(10**6))
	tcomp = pc*10*Sf*(D_oxtank+(2*tliner))/(2*6800*(10**6))
#	vliner = (np.pi*(D_oxtank**2/4)*L_oxtank)+(np.pi*4*(D_oxtank**3)/24)-(np.pi*((D_oxtank/2)-tliner)**2*(L_oxtank+(4*D_oxtank/6)))
#	vcomp = np.pi*(((D_oxtank/2)+tcomp)**2-(D_oxtank/2)**2)*(L_oxtank+(4*D_oxtank/6))
	Aliner = np.pi*((D_oxtank+(2*tliner))**2-(D_oxtank**2))/4
	mliner = Aliner*L_oxtank*4430
	Acomp =  np.pi*((D_oxtank+(2*tliner)+(2*tcomp))**2-(D_oxtank+(2*tliner))**2)/4
	mcomp = L_oxtank*1600*Acomp
	if Ox != 'H2O2':
	   mox = mliner+mcomp
	else:
	   mox = m_oxtank
	mf = m_futank+mli_futank
	Vol = V_oxtank + V_futank
	print(f'Tank volume: {Vol}')
	print(f"mf: {mf},mo: {m_oxtank}, mcomp: {mcomp}, mliner: {mliner},D_oxtank: {D_oxtank}, L_oxtank: {L_oxtank}, tliner: {tliner}, tcomp: {tcomp}, mox: {mox}")
	return V_oxtank, V_futank, mox, mf, res_fu, t_futank, Df, Lf

#PRESSURANT MASS CALCULATION
def Pressurization(Ox, mo, mf, V_oxtank, V_futank, code, mdot_ox, mdot_fu, rho_ox, rho_fu, pc):
	D_oxfeed = (4*mdot_ox/(np.pi*rho_ox*0.5))**(1/2)
	t_oxfeed = (pc*2*D_oxfeed*10)/(2*y_st)
	V_oxfeed = (np.pi*0.5/4)*((D_oxfeed+t_oxfeed)**2-D_oxfeed**2)
	if code != 'H':
	   D_fufeed = (4*mdot_fu/(np.pi*rho_fu*0.5))**(1/2)
	   t_fufeed = (pc*2*D_fufeed*10)/(2*y_st)
	   V_fufeed = (np.pi*0.5/4)*((D_fufeed+t_fufeed)**2-D_fufeed**2)
	   m_fufeed = V_fufeed*rho_st
	else:
	   m_fufeed = 0
	   D_fufeed = 0
	if code == 'H':
	   mp_fufeed = 0
	   if Ox != 'N2O':
	      V_pres = V_oxtank*ullage*5
	      mp_oxfeed = V_oxfeed*rho_st
	   else:
	      V_pres = 0
	      mp_oxfeed = 0
	elif code == 'D':
	   V_pres = 0
	   mp_oxfeed = 0
	   mp_fufeed = 0
	else:
	   V_pres = (V_oxtank+V_futank)*ullage*5
	   mp_oxfeed = V_oxfeed*rho_st
	   mp_fufeed = V_fufeed*rho_st
	m_He = (8e6*V_pres*M_He)/(R*297)
	D_Hetank = ((4*V_pres)/(3*np.pi))**(1/3)
	L_Hetank = 3*D_Hetank
	#A_Hetank = (np.pi*D_Hetank*L_Hetank)+(2*np.pi*(D_Hetank**2/4))
	t_Hetank = (3e7*D_Hetank)/(2*y_Al)
	A_Hetank = np.pi*((D_Hetank+t_Hetank)**2-D_Hetank**2)/4
	m_bottle = A_Hetank*t_Hetank*rho_Al
	m_oxfeed = V_oxfeed*rho_st
	m_feed = m_oxfeed+m_fufeed+mp_fufeed+mp_oxfeed
	m_pf = m_feed+m_bottle
	m_storage = mf+mo+m_feed
	#+m_He+m_bottle+m_feed
	print(f"mass_He:{m_He}, t: {t_Hetank},mass_bottle: {m_bottle}, mass_feed: {m_feed}, D_fufeed: {D_fufeed}, D_oxfeed: {D_oxfeed}")
	return m_storage, V_pres, m_pf, m_He

#NOZZLE MASS CALCULATION
def nozzle_Mass(At, Rt):
	Ae = At*AR
	De = 2*np.sqrt(Ae/np.pi)
	L_nozzle = ((Rt*(np.sqrt(AR)-1)+(1.5*Rt*(1/np.cos(15*np.pi/180)-1)))/np.tan(15*np.pi/180))*0.8
	#Nozzle case material Inconel 718
	t_nozzle = (pc*De*Sf)/(2*y_in718)
	D_outer = t_nozzle+De
	V_case = (np.pi/3)*((D_outer**2-De**2)/4)*L_nozzle
	m_case = V_case*rho_in718
	#Nozzle liner material ceramic
	V_liner = np.pi*De*L_nozzle*0.002
	m_liner = V_liner*rho_liner
	#Nozzle insulation using aerogel
	V_ins = np.pi*De*L_nozzle*0.004
	m_ins = V_ins*rho_ins
	V_nozzle = V_case+V_liner+V_ins
	m_nozzle = m_case+m_liner+m_ins
	print(f'm_nozzle: {m_nozzle}')
	return m_nozzle, V_nozzle, Ae

#COMBUSTION CHAMBER MASS CALCULATION
def chamber_Mass(pc, M_fu, M_ox, mdot_fu, mdot_ox, Tc, cstar, At, t_futank, Df_grain, Lf_grain, code):
	if code == 'H':
	   D_cc = Df_grain+res_fu+t_futank
	   L_postcc = Lf_grain*0.2
	   V_cc = (np.pi*((D_cc/2)**2)*L_postcc)+((4/3)*np.pi*((D_cc/2)**3))
	   m_cc = (pc*2*V_cc)/(9.81*15000)
	else:
	   M_gas = ((M_fu*mdot_fu)+(M_ox*mdot_ox))/(mdot_fu+mdot_ox)
	   R_sp = (R/M_gas)*1000
	   Lstar = (0.001*Tc*R_sp)/cstar
           #Cchamber wall material stainless steel 318
	   V_cc = Lstar*At
	   D_cc = ((4*V_cc)/(3*np.pi))**(1/3)
	   L_cc = 3*D_cc
	   t_cc = (pc*15*D_cc*Sf)/(2*y_in718)
	   V_outer = np.pi*((D_cc+(2*t_cc))/2)**2*L_cc
	   V_wall = V_outer-V_cc
	   m_wall = V_wall*rho_in718
	   V_outerins = np.pi*((D_cc+(2*t_cc)+(2*0.004))/2)**2*L_cc
	   #Cchamber insulation using aerogel
	   V_ins = V_outerins-V_cc
	   m_ins = V_ins*rho_ins
	   m_cc = m_ins+m_wall
	   V_cc += (V_wall+V_ins)
	print(f'm_cc: {m_cc}')
	return m_cc, V_cc, D_cc

#IGNITION SYSTEM MASS
def ignition_Mass(Ox, ig_code, Hv, Hv_f, Cp, delT, mdot_ox, mdot_fu, D_cc, code):
	if ig_code == 'H':
	   E_fu = 0
	   E_ox = 0
	elif code == 'H':
	   E_fu = 0.005*mdot_fu*Cp*delT
	else:
	   E_fu = 0.005*mdot_fu*Hv_f
	if Ox in ['H2O2', 'HTP90', 'HTP98']:
	   #catalytic ignition
	   V_bed = mdot_ox/245
	   L_bed = 0.1*mdot_ox
	   D_bed = 0.85*D_cc
	   t_bed = ((1.5e+6)*Sf*D_bed)/(2*y_in718)
	   m_bed = (rho_in718*L_bed*np.pi*((D_bed+t_bed)**2-(D_bed**2)))/4
	   #Catalyst assumed is MnO2 with density of 5000 kg/m3 and a porosity of 70%
	   #Porosity = 1-(V_cat/V_bed)
	   V_cat = (1-0.7)*V_bed
	   m_cat = V_cat*5000
	   m = m_bed+m_cat
	   V = V_bed+V_cat
	   E_ox = 0
	else:
	   E_ox = 0.005*mdot_ox*Hv
	   m = 0
	   V = 0
	#10 Capacitors with energy density of 5000 J/kg and time to ignite 0.5 sec
	if ig_code != 'H':
	   E_req = E_fu+E_ox
	   m_cap = E_req*5/(3.5*3600)
	   V_cap = np.pi*(0.002**2/4)*0.003*10
	   #Vol of Tungsten electrodes of Density 19250 kg/m3 with arc length 5mm, diameter 2mm
	   V_ele = np.pi*(0.002**2/4)*0.005
	   m_ele = 2*V_ele*19250
	   #Housing 0.015*0.005*0.010
	   V_house = 0.015*0.05*0.01
	   V_powerelec = 0.05*0.005*0.005
	   V_ig = V+V_cap+(2*V_ele)+V_house+V_powerelec
	   m_ig = m+m_cap+m_ele+0.5+(V_house*7850)
	else:
	   m_ig = 0
	   V_ig = 0
	   E_req = 0
	print(f"E_req: {E_req}, m_ig: {m_ig}")
	return m_ig, V_ig, E_req

#POWER SOURCE (SOLAR ARRAY AND BATTERRY) MASS CALCULATION
def powersource_Mass(ig_code, E_req, tb):
#	if (ig_code == 'H') or Ox in ['H2O2', 'HTP90', 'HTP98']:
#	   P_req = 1500
#	else:
	P_req = 1000+500
	   #GA-As Solar panels of Density 5320 kg/m3 with efficieny of 0.36, Inherent Degradation of 0.954838648874 
	flux_id = 1368*0.32
	flux_BOL = flux_id*0.954838648874*np.cos(40*np.pi/180)
	A_rho = 5320*0.0004
	A_sa = (0.05*P_req)/flux_BOL
	m_sa = A_sa*A_rho
	V_sa = m_sa/5320
	#Back up battery of Lithium-ion with efficiency 1.045 and DOD 0.35
	C_req = P_req*10/3600
	#C_EOL = C_req/(1.045*0.35)
	m_batt = C_req/180
	V_batt = C_req/300000
	m_power = m_sa+m_batt
	V_power = V_sa+V_batt
	print(f'm_sa: {m_sa}, m_batt: {m_batt}')
	return m_power, V_power

#FINAL PERFORMANCE DATA
def performance(Ivac, mdot, Ae, Wet_mass, mf, V_env, rho_ox, rho_fu, OF, Pe):
	Thrust = (Ivac*mdot)+((Pe-0)*Ae)
	DeltaV = Ivac*np.log(Wet_mass/mf)
	rho_Dv = DeltaV*(V_env/Wet_mass)
	rho_p = (rho_ox*rho_fu*(1+OF))/((rho_fu*OF)+rho_ox)
	Vol_Isp = Ivac*rho_p
	return Thrust, DeltaV, rho_Dv, Vol_Isp

if __name__ == "__main__":
	Fu, Ox, Pc, Rt, tb, mp = opts(sys.argv[1:])
	pc = Pc*(10**5)
	output = subprocess.check_output(f'python3 {path} -f {Fu} -o {Ox} -p {Pc}', shell=True)
	op = eval(output.decode("utf-8").strip())
	if op['Ivac'] == None:
	   print('Inputs out of range, Provide Valid inputs.\nThe available datasets are:[Fuel- (HTPB, HDPE, Paraffin, RP1, MMH, UDMH, N2H4, LH2, CH4, Aerozine50), Oxidizers - LOX, N2O, H2O2, HTP90, HTP98, HNO3, OF2, N2O4]')
	else:
	   Hybrid = ['HDPE', 'HTPB', 'Paraffin']
	   if Fu in Hybrid:
	      code = 'H'
	   elif Fu == 'C3H6':
	      code = 'D'
	   else:
	      code = 'L'
	   if (Fu in ['MMH', 'UDMH', 'Aerozine50'] or Ox == 'HNO3'):
	      ig_code = 'H'
	   elif (Fu == 'N2H4' and Ox == 'N2O4'):
	      ig_code = 'H'
	   else:
	      ig_code = 'I'
	   Dt = 2*Rt
	   At = np.pi*(Rt**2)
	   mdot = pc*At/(op['cstar']*0.98)
	   mdot_ox, mdot_fu, m_fu, m_ox = prop_Mass(mdot, op['OF'], tb, Ox, Fu)
	   V_oxtank, V_futank, m_oxtank, m_futank, res_fu, t_futank, Df, Lf = tank_Mass(pc, Rt, tb, mdot_ox, m_fu, m_ox, op['rho_ox'], op['rho_fu'], code, op['n'], op['a'])
	   m_storage, V_pres, m_pf, m_He = Pressurization(Ox, m_oxtank, m_futank, V_oxtank, V_futank, code, mdot_ox, mdot_fu, op['rho_ox'], op['rho_fu'], pc)
	   m_nozzle, V_nozzle, Ae = nozzle_Mass(At, Rt)
	   m_cc, V_cc, D_cc = chamber_Mass(pc, op['M_fu'], op['M_ox'], mdot_fu, mdot_ox, op['Tc'], op['cstar'], At, t_futank, Df, Lf, code)
	   m_ig, V_ig, E_req = ignition_Mass(Ox, ig_code, op['Hv'], op['Hv_f'], op['Cp'], op['delT'], mdot_ox, mdot_fu, D_cc, code)
	   m_power, V_power = powersource_Mass(ig_code, E_req, tb)
	   propellant_mass = m_fu+m_ox
	   print(f"m_storage: {m_storage}, m_nozzle: {m_nozzle}, m_cc: {m_cc}, m_power: {m_power}")
	   Prop_Dry_mass = m_storage+m_nozzle+m_cc+m_ig+m_power
	   Dry_mass = Prop_Dry_mass+mp+49
	   Wet_mass = Dry_mass+propellant_mass
	   mfinal = Dry_mass+(0.15*propellant_mass)
	   V_env = (1.2*(V_oxtank+V_futank+V_pres))+V_nozzle+V_cc+V_ig+V_power
	   F, DeltaV, rho_Dv, Vol_Ivac = performance(op['Ivac'], mdot, Ae, Wet_mass, mfinal, V_env, op['rho_ox'], op['rho_fu'], op['OF'], op['Pe'])
	   print(f"Ivac: {op['Ivac']}, Thrust (N): {F}\n Delta-V (m/s): {DeltaV}\n Density Delta-V: {rho_Dv}\n Volumetric Isp: {Vol_Ivac}\n Propellant Mass: {propellant_mass}\n Propulsion Structure mass: {Prop_Dry_mass}")

	   # Define the object's density (kg/m³)
	   rho =  1000 # Example value

	   # Define the limits of integration (object bounding box)
	   x_min, x_max = -1, 1  # meters
	   y_min, y_max = -1, 1
	   z_min, z_max = -1.5, 1.5

	   r = Rt * 25
	   h = 3 * r
	   # Define the shape function
	   #def inside_shape(x, y, z):
	    #  """Return True if (x, y, z) is inside the object, False otherwise."""
	      # For now, assume the entire bounding box is filled (a cube)
	     # return True

	   # Define the integrands for moments of inertia
	   def integ_Ixx():
	      #if inside_shape(x, y, z):
	      Ixx_wet = (1/12) * Wet_mass*(3*r**2 + h**2)
	      Ixx_dry = (1/12) * mfinal*(3*r**2 + h**2)
	      return Ixx_wet, Ixx_dry
	   def integ_Iyy():
	      Iyy_wet = (1/12) * Wet_mass*(3*r**2 + h**2)
	      Iyy_dry = (1/12) * mfinal*(3*r**2 + h**2)
	      return Iyy_wet, Iyy_dry
	   def integ_Izz():
	      Izz_wet = (1/2) * Wet_mass * r**2
	      Izz_dry = (1/2) * mfinal * r**2
	      return Izz_wet, Izz_dry
	      #else:
                #return 0

	   # Set up the integration
	   #bounds = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

	   # Compute Ixx
	   Ixx_wet, Ixx_dry = integ_Ixx()

	   # Compute Iyy
	   #Iyy, err_yy = integrate.nquad(integrand_Iyy, bounds)
	   Iyy_wet, Iyy_dry = integ_Iyy()

	   # Compute Izz
	   #Izz, err_zz = integrate.nquad(integrand_Izz, bounds)
	   Izz_wet, Izz_dry = integ_Izz()
	   print(f"Ixx_wet = {Ixx_wet:.4f}, Ixx_dry = {Ixx_dry:.4f} kg·m²")
	   print(f"Iyy_wet = {Iyy_wet:.4f}, Iyy_dry = {Iyy_dry:.4f} kg·m²")
	   print(f"Izz_wet = {Izz_wet:.4f}, Ixx_dry = {Izz_dry:.4f} kg·m²")

