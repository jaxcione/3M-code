import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from scipy.special import pbdv, pbvv
import math
import warnings

warnings.filterwarnings("ignore")

"""
Project 1: SS Solution for a Parallel Plate Model
Authors: Jaxon, Chuan Heng, Andrew Hickman

Our Governing PDE:dC/dt+ u(y) dC/dx = D_A d²C/dy

Velocity profile: u(y) = alpha * y*(y-H),  alpha<0  ->  u(y)>0 for y in (0,H)
IC: C(x,y,0)=Ca0
BCs: D_A dC/dy|_{y=0}  = k C|_{y=0}    (first-order reactive bottom wall)
        dC/dy|_{y=H/2}    = 0             (symmetry at centreline)
        C(x=0, y, t)      = CA0           (uniform inlet, step at t=0)
Full solution: C(x,y,t) = C_SS(x,y) + C_transient(x,y,t)
    where C_SS(x,y) = Sum[ A_n_SS*psi_n_SS(y)*exp(-lam2_n*x) ] (steady state solution) and C_transient(x,y,t) = Sum[ A_n * psi_n(y) * exp(-mu_n * x) * exp(-lam_n * t) ] (transient solution)

"""

# ─── Parameters(given)
R=8.314 #J/(mol*K)
E_a=40000 #J/mol
T=300 #k subject to change based on user input

k=.02 #assuming a faster rxn for as of now
# k_0=2000
# k=k_0*math.exp(-E_a/(R*T)) #rate constant


alpha= -400 #deltaP/2visocsityL (negative to ensure forward flow in u(y)=2alpha(y^2-Hy))
D_A=1E-5 #diffusion coeff  m^2/s of air
CA0=.0174 #mol/m^3 .. ambient condition
H=.05    #m height of plate
L=1 #m length of the bed

#velcity profile for parabolic flow(derived from Navier-Stokes for laminar flow between parallel plates)
def velocity_profile(y):
    return alpha * y * (y - H)#see slides

#SS SOLUTION-----------------------------------------------------------------------------------------------------------------------------
#finding eigen value ss
def det_ss(lam2):
    #substitions we made(see slide) 
    kappa = abs(alpha) / D_A 
    param = (4 * lam2 * kappa)**0.25 #param is omega in the slides

    v = (H**2 / 8) * math.sqrt(kappa * lam2) - 0.5
    z0 = -param * H / 2 #plugging in y=0 for bottom wall boundary condition
    z1 = 0.0 #at centerline

    U0, dU0 = pbdv(v, z0)# parabolic cylinder functions and their derivatives at the bottom wall for BC evaluation
    V0, dV0 = pbvv(v, z0)
    _, dU1 = pbdv(v, z1) #dervivies at centerline for symmetry condition
    _, dV1 = pbvv(v, z1)
    
    if abs(dU1) < 1e-18 or abs(dV1) < 1e-18 or not np.isfinite(dU1*dV1): #ensuring we don't divide by zero or encounter numerical issues
        return np.nan
    
    ratio = -dV1 / dU1 #ratio i showed in the slides
    return param * (ratio * dU0 + dV0) + (k/D_A) * (ratio * U0 + V0) #this is F(lam) . Note since this is not the transient we only have one eigen value

#this function returns the analytical form of the ode w repsect to y, thus parapbolic cylinder functions
#i said here in the code psi is the parabolic cylinder function solution to the ODE in y 
def psi_ss(y, lam2):
    kappa = abs(alpha) / D_A
    param = (4 * lam2 * kappa)**0.25
    v = ((H**2) / 8) * math.sqrt(kappa * lam2) - 0.5
    
    # Use symmetry: evaluate at distance from centerline
    y_eff = min(y, H - y) #doing this to map symetrically
    z = param * (y_eff - H/2) 
    z1 = 0.0 #at centerline for symmetry condition
    
    U, _ = pbdv(v, z) #parabolic cylinder functions at the point of interest, or y, for constructing the eigenfunction profile
    V, _ = pbvv(v, z) #parabolic cylinder functions at the point of interest or y, for constructing the eigenfunction profile
    _, dU1 = pbdv(v, z1) #derivative at centerline for symmetry condition
    _, dV1 = pbvv(v, z1) #derivative at centerline for symmetry condition
    
    if abs(dU1) < 1e-18: #ensuring we don't divide by zero or encounter numerical issues
        return 0.0
    
    ratio = -dV1 / dU1 #same ratio we contrived in the slides but for SS case
    psi_half = ratio * U + V #mirroring across centerline to get full profile, but since we only compute for y in [0,H/2], this gives us the correct value for the entire range due to symmetry
    
    return psi_half

print("getting lamda")
lam2_scan = np.linspace(0.1, 12000, 10000) #making an array of all lam values
roots_ss = [] #storing them here
for i in range(len(lam2_scan)-1): #scanning through the lam values to find sign changes in det_ss which indicate eigenvalues
    a, b = lam2_scan[i], lam2_scan[i+1]
    fa = det_ss(a) #solving for the derminant at the endpoints of the interval to check for sign changes
    fb = det_ss(b)
    if np.isnan(fa) or np.isnan(fb) or not np.isfinite(fa*fb):
        continue
    if fa * fb < 0: #if there is a sign change, we know there is a root in the interval, so we use brentq 
        try:
            rt = brentq(det_ss, a, b, xtol=1e-9) #root finding w brentq to get the zeros
            if abs(det_ss(rt)) < 1e-3:
                roots_ss.append(rt) #storing them here
        except:
            pass

roots_ss = np.sort(np.unique(np.round(roots_ss, decimals=8))) # getting all unque roots 

#getting the coefficients for the SS solution using orthogonality and inlet condition
y_grid_half = np.linspace(0, H/2, 1200)
u_grid_half = velocity_profile(y_grid_half)

A_ss = []
for lam2 in roots_ss[:50]:  # limit to reasonable number of modes which in this case is 50
    psi_raw = np.array([psi_ss(y, lam2) for y in y_grid_half])
    psi_raw[~np.isfinite(psi_raw)] = 0.0
    
    num = simpson(u_grid_half * CA0 * psi_raw, y_grid_half) #using orthogonality and inlet condition to get the coefficients for the SS solution
    den = simpson(u_grid_half * psi_raw**2, y_grid_half)
    if abs(den) < 1e-18 or not np.isfinite(den): #making sure we dont divide by zero
        A_ss.append(0.0)
    else:
        A_ss.append(num / den)
A_ss = np.array(A_ss) #storing as a numpy array for easier manipulation later

# full ss function 
def CA_ss(x, y):#summing over all modes to get the full SS solution at any point (x,y)
    total = 0.0 
    for A, lam2 in zip(A_ss, roots_ss):
        total += A * psi_ss(y, lam2) * np.exp(-lam2 * x) #this is thge fourier series pretty mcuh
    return total

#testing y values to see if we get the correct inlet condition and reasonable profile
y_test = [0.0, H*0.1, H*0.25, H*0.5, H*0.75, H]
for y in y_test:
    c = CA_ss(0, y)
    print(f"y={y:.4f} → C={c:.6f}  (target {CA0:.6f})  error={(c-CA0)/CA0*100:+.1f}%") #seeing errors. If there are, this is due to truncation of modes 



x_vals= np.linspace(0,L,70)# Grid for spatial map (full height)
y_vals = np.linspace(0,H,80)      # now full 0 to H
X, Y = np.meshgrid(x_vals,y_vals)

t_snap=0.0   
Z = np.zeros_like(X)

for i in range(len(y_vals)):
    for j in range(len(x_vals)):
        Z[i,j] = CA_ss(x_vals[j], y_vals[i]) / CA0

# Time series at exit centerline
t_vals = np.linspace(0, 500, 120)
exit_conc = np.array([CA_ss(L, H/2) / CA0 for _ in t_vals])  # constant with SS only

fig = plt.figure(figsize=(14, 6))


ax1 = fig.add_subplot(1, 1, 1)
cp = ax1.contourf(X, Y, Z, levels=40, cmap='viridis')
fig.colorbar(cp, ax=ax1, label='$C_A / C_{A0}$')
ax1.axvline(x=L, color='white', ls='--', lw=1.2, alpha=0.7, label='Exit')
ax1.set_title(f'Spatial Concentration Profile\n(full channel, t = {t_snap:.1f} s)')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.legend(loc='upper right')
plt.tight_layout()
plt.show()