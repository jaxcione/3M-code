import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from scipy.interpolate import interp1d

"""
Authors: Jaxon Cione, Chuan Heng, and Andrew Hickman
Date: 2/28/2026
Description: This code solves the PDE for a 2D adsorption process using finite volume/ Tridiagonal matrix method. Gas phase is treated as
psuedo-steady at each timestep. Which we found to be a reasonable assumption after talking with our fluid dynamics professor and from "solving" the semi-analytic solution
"""

#parameters------------------------------------------------------------------
mw_CO2=44.01/1000 #kg/mol, molecular weight of CO2
ms_pp=1 #kg/m^2 of sorbent
rho=1000 #kg/m^3, density of sorbent. We don't know this but we can change it later
R=8.314 #J/(mol*K)
C_0=.017 #mol/m^3... ambient CO2 conecentration .01725
viscosity=1.81E-5 #of air at room temp, kg/m/s
D_A=1.6E-5 #diffusion coeff m^2/s
L=1 #m length of the bed
H=.01 #m height of the plate

#values given from Neeraj ------------------------------------------------------------------
tau=.4
k_0=2000 #1/s
Xi=.7
D_H=70000 #J/mol delta H of adsorption
E_a=40000 #J/mol #activation energy for adsorption
T_0=298.15 #K 
q_m0=3 #mol-CO2/kg
b_0=2E-14 #1/Pa
T=298.15 #K, subject to change based on user input

k=k_0*np.exp(-E_a/(R*T)) #rate constant
b=b_0*np.exp(D_H/(R*T))
q_m=(q_m0)*(np.exp(Xi*(1-T/T_0))) #sorbent isotherm
#Gridsize and snapshots----------------------------------------------------------
#note we are doing finite volume so doing a N_x X N_y grid

N_y=30 #number of grid points in y direction 15
N_x=100#number of grid points in x direction, 60 


dx=L/N_x #grid spacing in x direction
dy=(H/2)/N_y #grid spacing in y direction(half the domain)
x=np.linspace(dx/2,L-dx/2,N_x) #x coordinates of cell centers
y_coord=np.linspace(dy/2,H/2-dy/2,N_y) #y coordinates of

#creating arrays and initializing functions----------------------------------------------------------------
y0 = np.concatenate([ np.full((N_x, N_y), C_0).flatten(),np.zeros(N_x),np.zeros(N_x)])

USE_LINEAR =False#turn off if we dont want non linear isotherm 
K_H=q_m*(b*R*T)**tau
def q_e_vec(C):
    if USE_LINEAR:
        return K_H*np.maximum(C,0)
    else:
        brtc = np.maximum(b*R*T*C,0)
        return q_m*brtc/(1+brtc**tau)**(1/tau) #from neeraj 
#velocity stuff----------------------------------------------------------------

v_avg_target=10 #user input m/s

alpha = -v_avg_target*6/(H**2)  #alpha=(deltaP)/(2*mu*L) where deltaP is the pressure drop across the bed, mu is viscosity, and L is length. We can adjust alpha to get the desired v_avg
def velocity(y):
    return alpha *(y**2-H*y)
v_profile =(velocity(y_coord))
print("alpha =", alpha)



#Solving right hand side via finite volume method------------------------------------------------------------
#right had side of the ODE system, will be used in the ODE solver qb is bottom loading qt is top loading.
def C_ss(qb):
    C = np.zeros((N_x, N_y))
    beta_j = (D_A * dx) / (v_profile * dy**2)

    # ---- Inlet column (i=0) ----
    rhs = np.full(N_y, C_0)
    banded = np.zeros((3, N_y))

    # Interior rows
    banded[1, :] = 1 + 2*beta_j
    banded[0, 1:] = -beta_j[:-1]     # FIX: was -beta_j[1:]
    banded[2, :-1] = -beta_j[1:]     # FIX: was -beta_j[:-1]

    # Bottom BC (ghost cell, consistent)
    gamma_b = (k * ms_pp * dy / D_A) * (q_e_vec(C_0) - qb[0])
    banded[1, 0] = 1 + 2*beta_j[0]   # FIX: was 1 + beta_j[0]
    banded[0, 1] = -2*beta_j[0]      # FIX: ghost cell doubles upper coeff at j=0
    rhs[0] -= 2*beta_j[0] * gamma_b

    # Top BC (symmetry, dc/dy=0)
    banded[1, -1] = 1 + beta_j[-1]

    C[0] = solve_banded((1, 1), banded, rhs)

    #columns that are after inlet. Neeraj helped by providing better boundary conditions for the inlet 
    for i in range(1, N_x):
        Cprev = C[i-1]
        
        rhs = Cprev.copy()
        beta_j = (D_A * dx) /(v_profile*dy**2) # beta as seen in derivation

        banded_matrix = np.zeros((3,N_y)) #3xN_y because we are getting rid of all non-zero entries of the matrix and only storing the actual values 
        
        banded_matrix[1,:]=1+2*beta_j #second row ion banded
        banded_matrix[0, 1:] = -beta_j[:-1]   #first rw in banded
        banded_matrix[2, :-1] = -beta_j[1:]   #last row in banded

        # Bottom BC doing ghost cell method
        gamma_b = (k * ms_pp*dy / D_A) * (q_e_vec(Cprev[0]) - qb[i])
        banded_matrix[1, 0]=1+2*beta_j[0]  
        banded_matrix[0, 1]=-2*beta_j[0]     
        rhs[0] -= 2*beta_j[0]*gamma_b

        # Top BC by symmetry 
        banded_matrix[1, -1]=1+beta_j[-1]

        C[i] = solve_banded((1, 1),banded_matrix, rhs)
    return np.clip(C, 0, C_0) #clipping s.t we dont have unphysical values


dt=5#time step in seconds
t_f=36000 #final time 10 hours... this is a long time. but we want to check the long term behavior of our model vs comsols 
N_t=int(t_f/dt)+1 #number of time steps, we will use this to create our time array for the ODE solver.
t_array=np.linspace(0,t_f,N_t) #time array for ODE solver.

qb=np.zeros(N_x) #initializing bottom loading array, this will be used in the ODE solver.

C_at_time_snaps=[] #this will store the concentration profiles at the time snapshots we want to plot later.
CO2_captured=[] #this will store the total mass of CO2 captured at each time step
qb_hist=[] #for plotting
C_outlet_avg=[] #for plotting, this will store the average concentration at the outlet for

time_snaps = [0,7200,14400,18000,36000] #time snapshots. For plotting
steps_at_time_snaps=[int(t/dt) for t in time_snaps] #the time steps that correspond to the time snapshots. Allows for shorter computation time

def dqdt(t,q,qe): #ODE in time 
    return k*(qe-q)

for n in range(N_t):
    C=C_ss(qb) #solving for the concentration profile at the current time step.
    qe_b = q_e_vec(C[:,0]) #this is at the bottom of the plate
    sol_b=solve_ivp(fun=lambda t, q: dqdt(t, q, qe_b), t_span=[0, dt], y0=qb, method='RK45')#solve bottom loading
    
    qb=sol_b.y[:,-1] #updating bottom loading to be the value at the end of the time step
    
    #copying the loading profiles to store for plotting later
    qb_hist.append(qb.copy())

    if n in steps_at_time_snaps: #if the current time is in our snapshot times, we will store the concentration profile for plotting.
        C_at_time_snaps.append(C.copy()) #storing concentration profile for plotting.

    q_total_current=qb
    C_adsorbed_current = np.sum(q_total_current *ms_pp*(dx))
    CO2_captured.append(C_adsorbed_current)
    C_outlet_avg.append(C[-1, :].mean()) #grabbing the mean for plotting

C_snaps = {}
C = C_ss(qb) #solving one last time
C_snaps[36000]= C.copy() #storing a copy of all 10 hours

q_total=qb*2#total loading from top and bottom hence multiplication by 2 #assuming symettry 
q_model=q_total*(ms_pp) #mol/m^2 will be comparing with comsol

# validation ---------------------------------------------------------------------------------------------
comsol_filepath = "3M//validation//COMSOL_Data//hourly.txt" #this is a relative path 
segments = []
current_x, current_y = [], [] #initializing x,y arrays 

with open(comsol_filepath,'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        parts=line.split()
        if len(parts)>= 2:
            try:
                xval=float(parts[0])
                yval=float(parts[1])
            except ValueError:
                continue
            # X reset detected: if current segment is non-empty and x jumps back near 0
            if current_x and xval < current_x[-1] - 0.5:
                segments.append((np.array(current_x), np.array(current_y)))
                current_x, current_y = [], []
            current_x.append(xval)
            current_y.append(yval)
if current_x:
    segments.append((np.array(current_x), np.array(current_y)))
n_seg = len(segments) #number of segtments within the hourly data

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10), sharex=True)
#color gradients
cmap_comsol = plt.cm.plasma 
cmap_python  = plt.cm.viridis
 
model_hours = list(range(0, n_seg))
 
for i, (xc, yc) in enumerate(segments):
    color = cmap_comsol(i / (n_seg - 1))
    step_ds = max(1, len(xc) // 300)
    ax1.plot(xc[::step_ds], yc[::step_ds],
             color=color, linewidth=1.8,
             label=f'COMSOL {i} hr')
 
for i, hour in enumerate(model_hours):
    color = cmap_python(i/(n_seg -1)) #assigning the color for each segment
    idx= min(int(hour*3600/dt), len(qb_hist) -1)# converting to hours
    q_t =qb_hist[idx] *ms_pp #q total for one palste
    ax1.plot(x, q_t,
             color=color, linewidth=1.8, linestyle='--',
             label=f'Python {hour} hr')
 
 #plotting n stuff
ax1.set_ylabel('CO₂ Adsorbed (mol/m²)', fontsize=13)
ax1.set_title('CO₂ Adsorbed Along Active Surface — Model vs. COMSOL', fontsize=13)
ax1.ticklabel_format(axis='y',style='sci', scilimits=(0, 0))
ax1.grid(True, linestyle='--',alpha=0.5)
ax1.set_xlim(left=0)
ax1.legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.show()