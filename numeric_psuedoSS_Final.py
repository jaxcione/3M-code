import numpy as np
import math 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from matplotlib.ticker import MultipleLocator


"""
    Authors: Jaxon Cione, Chuan Heng, and Andrew Hickman
    Date: 2/28/2026
    Description: This code solves the PDE for a 2D adsorption process using finite volume/ Tridiagonal matrix method. Gas phase is treated as
    psuedo-steady at each timestep. Which we found to be true after talking with our fluid dynamics professor/ "solving" semi-analytic solution   
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
N_x=100 #number of grid points in x direction, 60 


dx=L/N_x #grid spacing in x direction
dy=(H/2)/N_y #grid spacing in y direction(half the domain)
x=np.linspace(dx/2,L-dx/2,N_x) #x coordinates of cell centers
y_coord=np.linspace(dy/2,H/2-dy/2,N_y) #y coordinates of

#creating arrays and initializing functions----------------------------------------------------------------
y0 = np.concatenate([ np.full((N_x, N_y), C_0).flatten(),np.zeros(N_x),np.zeros(N_x)])

def q_e_vec(C):
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
        banded_matrix[0, 1:] =-beta_j[:-1]   #first rw in banded
        banded_matrix[2,:-1] =-beta_j[1:]   #last row in banded

        #bottom BC's
        gamma_b =(k*ms_pp*dy/D_A)*(q_e_vec(Cprev[0])-qb[i]) #defining gamma to be the flux terms
        banded_matrix[1, 0]=1+2*beta_j[0]  
        banded_matrix[0, 1]=-2*beta_j[0]     
        rhs[0]-= 2*beta_j[0]*gamma_b

        #top usiny symmetry
        banded_matrix[1, -1]=1+beta_j[-1]

        C[i] = solve_banded((1, 1),banded_matrix, rhs)

    return np.clip(C, 0, C_0) #clipping s.t we dont have unphysical values


dt=2#time step in seconds
t_f=36000 #final time
N_t=int(t_f/dt)+1 #number of time steps, we will use this to create our time array for the ODE solver.
t_array=np.linspace(0,t_f,N_t) #time array for ODE solver.

qb=np.zeros(N_x) #initializing bottom loading array, this will be used in the ODE solver.

C_at_time_snaps=[] #this will store the concentration profiles at the time snapshots we want to plot later.
CO2_captured=[] #this will store the total mass of CO2 captured at each time step
qb_hist=[] #for plotting
C_outlet_avg=[] #for plotting, this will store the average concentration at the outlet for
qb_hist_snaps=[]

time_snaps = [0,1800,3600,7200,14400,36000] #time snapshots. For plotting
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
        qb_hist_snaps.append(qb.copy())

  
    q_total_current=qb*2
    C_adsorbed_current = np.sum(q_total_current *ms_pp*(dx))
    CO2_captured.append(C_adsorbed_current)
    C_outlet_avg.append(C[-1, :].mean()) #grabbing the mean for plotting

C_snaps = {}
C = C_ss(qb) #solving one last time
C_snaps[36000]= C.copy() #storing a copy of all 10 hours

q_total=qb*2*ms_pp#total loading from top and bottom hence multiplication by 2 #assuming symettry 

"Note:I had AI plot this. I beieve ai has superior graphing abilities than me. Nonetheless, I still reviewed the code and made sure there was no errors"
#Plotting--------------------------------------------------------------------

# ── Style config ────────────────────────────────────────────────────────────
# ── Style config ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#dddddd",
    "axes.labelcolor":   "#333333",
    "axes.titlecolor":   "#111111",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "text.color":        "#333333",
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CMAP_HM = "plasma"

x_axis = np.linspace(0, L, N_x)
y_full = np.concatenate([y_coord, H - y_coord[::-1]])
X_full, Y_full = np.meshgrid(x_axis, y_full, indexing='ij')
C_outlet_avg = np.array(C_outlet_avg)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=10, fontweight='semibold')
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.grid(True, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

# ── 1. Heatmaps ───────────────────────────────────────────────────────────────
snap_labels = ["0 hr", "0.5 hr", "1 hr", "2 hr", "4 hr", "10 hr"]
print(len(C_at_time_snaps))  
for i, C_plot in enumerate(C_at_time_snaps):
    C_full = np.concatenate([C_plot, C_plot[:, ::-1]], axis=1)
    C_norm = np.clip(C_full / C_0, 0, 1)

    fig, ax = plt.subplots(figsize=(11, 3.0))
    cont = ax.contourf(X_full, Y_full * 1000, C_norm,
                       levels=np.linspace(0, 1, 50), cmap=CMAP_HM)
    cbar = fig.colorbar(cont, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("C / C₀", labelpad=8, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(f"Gas-Phase Concentration  —  t = {snap_labels[i]}",
                 fontweight='semibold', pad=9)
    ax.set_xlabel("Axial Position  [m]", labelpad=6)
    ax.set_ylabel("Gap  [mm]", labelpad=6)
    ax.tick_params(which='both', direction='out')
    plt.tight_layout()
    plt.show()

# ── 2. Breakthrough curve ─────────────────────────────────────────────────────

import numpy as np
import math 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from matplotlib.ticker import MultipleLocator


"""
    Authors: Jaxon Cione, Chuan Heng, and Andrew Hickman
    Date: 2/28/2026
    Description: This code solves the PDE for a 2D adsorption process using finite volume/ Tridiagonal matrix method. Gas phase is treated as
    psuedo-steady at each timestep. Which we found to be true after talking with our fluid dynamics professor/ "solving" semi-analytic solution   
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
N_x=100 #number of grid points in x direction, 60 


dx=L/N_x #grid spacing in x direction
dy=(H/2)/N_y #grid spacing in y direction(half the domain)
x=np.linspace(dx/2,L-dx/2,N_x) #x coordinates of cell centers
y_coord=np.linspace(dy/2,H/2-dy/2,N_y) #y coordinates of

#creating arrays and initializing functions----------------------------------------------------------------
y0 = np.concatenate([ np.full((N_x, N_y), C_0).flatten(),np.zeros(N_x),np.zeros(N_x)])

def q_e_vec(C):
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
        banded_matrix[0, 1:] =-beta_j[:-1]   #first rw in banded
        banded_matrix[2,:-1] =-beta_j[1:]   #last row in banded

        #bottom BC's
        gamma_b =(k*ms_pp*dy/D_A)*(q_e_vec(Cprev[0])-qb[i]) #defining gamma to be the flux terms
        banded_matrix[1, 0]=1+2*beta_j[0]  
        banded_matrix[0, 1]=-2*beta_j[0]     
        rhs[0]-= 2*beta_j[0]*gamma_b

        #top usiny symmetry
        banded_matrix[1, -1]=1+beta_j[-1]

        C[i] = solve_banded((1, 1),banded_matrix, rhs)

    return np.clip(C, 0, C_0) #clipping s.t we dont have unphysical values


dt=2#time step in seconds
t_f=36000 #final time
N_t=int(t_f/dt)+1 #number of time steps, we will use this to create our time array for the ODE solver.
t_array=np.linspace(0,t_f,N_t) #time array for ODE solver.

qb=np.zeros(N_x) #initializing bottom loading array, this will be used in the ODE solver.

C_at_time_snaps=[] #this will store the concentration profiles at the time snapshots we want to plot later.
CO2_captured=[] #this will store the total mass of CO2 captured at each time step
qb_hist=[] #for plotting
C_outlet_avg=[] #for plotting, this will store the average concentration at the outlet for
qb_hist_snaps=[]

time_snaps = [0,1800,3600,7200,14400,36000] #time snapshots. For plotting
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
        qb_hist_snaps.append(qb.copy())

  
    q_total_current=qb*2
    C_adsorbed_current = np.sum(q_total_current *ms_pp*(dx))
    CO2_captured.append(C_adsorbed_current)
    C_outlet_avg.append(C[-1, :].mean()) #grabbing the mean for plotting

C_snaps = {}
C = C_ss(qb) #solving one last time
C_snaps[36000]= C.copy() #storing a copy of all 10 hours

q_total=qb*2*ms_pp#total loading from top and bottom hence multiplication by 2 #assuming symettry 

"Note:I had AI plot this. I beieve ai has superior graphing abilities than me. Nonetheless, I still reviewed the code and made sure there was no errors"
#Plotting--------------------------------------------------------------------

# ── Style config ────────────────────────────────────────────────────────────
# ── Style config ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#dddddd",
    "axes.labelcolor":   "#333333",
    "axes.titlecolor":   "#111111",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "text.color":        "#333333",
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.8,
    "font.family":       "sans-serif",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CMAP_HM = "plasma"

x_axis = np.linspace(0, L, N_x)
y_full = np.concatenate([y_coord, H - y_coord[::-1]])
X_full, Y_full = np.meshgrid(x_axis, y_full, indexing='ij')
C_outlet_avg = np.array(C_outlet_avg)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=10, fontweight='semibold')
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.grid(True, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

# ── 1. Heatmaps ───────────────────────────────────────────────────────────────
snap_labels = ["0 hr", "0.5 hr", "1 hr", "2 hr", "4 hr", "10 hr"]
print(len(C_at_time_snaps))  
for i, C_plot in enumerate(C_at_time_snaps):
    C_full = np.concatenate([C_plot, C_plot[:, ::-1]], axis=1)
    C_norm = np.clip(C_full / C_0, 0, 1)

    fig, ax = plt.subplots(figsize=(11, 3.0))
    cont = ax.contourf(X_full, Y_full * 1000, C_norm,
                       levels=np.linspace(0, 1, 50), cmap=CMAP_HM)
    cbar = fig.colorbar(cont, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("C / C₀", labelpad=8, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(f"Gas-Phase Concentration  —  t = {snap_labels[i]}",
                 fontweight='semibold', pad=9)
    ax.set_xlabel("Axial Position  [m]", labelpad=6)
    ax.set_ylabel("Gap  [mm]", labelpad=6)
    ax.tick_params(which='both', direction='out')
    plt.tight_layout()
    plt.show()

# ── 2. Breakthrough curve ─────────────────────────────────────────────────────

C_outlet_norm = (C_outlet_avg - C_outlet_avg[0]) / (C_0 - C_outlet_avg[0])
t_hrs = t_array[:len(C_outlet_norm)] / 3600 # convert to hours

# find breakthrough and saturation times
bt_thresh  = 0.05
sat_thresh = 0.95

bt_idx  = np.argmax(C_outlet_norm >= bt_thresh)
sat_idx = np.argmax(C_outlet_norm >= sat_thresh)

t_bt  = t_hrs[bt_idx]  if bt_idx  > 0 else None
t_sat = t_hrs[sat_idx] if sat_idx > 0 else None

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(t_hrs, C_outlet_norm, linewidth=2, color="#4e79a7", label="$C_{out}$ / $C_0$")
ax.axhline(bt_thresh,  color="#e05c3a", linewidth=1.2, linestyle="--",
           label=f"Breakthrough threshold ({bt_thresh})")
ax.axhline(sat_thresh, color="#59a14f", linewidth=1.2, linestyle="--",
           label=f"Saturation threshold ({sat_thresh})")

# annotate breakthrough and saturation times
if t_bt is not None:
    ax.axvline(t_bt,  color="#e05c3a", linewidth=0.8, linestyle=":")
    ax.text(t_bt + 0.05, 0.08, f"$t_{{bt}}$ = {t_bt:.1f} hr",
            color="#e05c3a", fontsize=9)

if t_sat is not None:
    ax.axvline(t_sat, color="#59a14f", linewidth=0.8, linestyle=":")
    ax.text(t_sat + 0.05, 0.88, f"$t_{{sat}}$ = {t_sat:.1f} hr",
            color="#59a14f", fontsize=9)

style_ax(ax, "Breakthrough Curve — Outlet Concentration vs Time",
         "Time  [hr]", "$C_{out}$ / $C_0$")
ax.set_xlim(0, t_hrs[-1])
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
# ── 3. Sorbent loading vs time ────────────────────────────────────────────────
time_axis       = np.linspace(0, t_f, len(qb_hist))
loading_vs_time = [np.mean(q) * ms_pp * 2 for q in qb_hist]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_axis / 3600, loading_vs_time, linewidth=2, color="#2ca02c")

style_ax(ax, "Spatially Averaged Sorbent Loading vs Time",
         "Time  [hr]", "Loading  [mol m$^{-2}$]")
plt.tight_layout()
plt.show()

# ── 4. Total CO₂ captured ─────────────────────────────────────────────────────
W = 1.0
CO2_captured_3d = [val * W for val in CO2_captured]

fig, ax = plt.subplots(figsize=(8, 4.5))
t_cap = t_array[:len(CO2_captured_3d)] / 3600

ax.plot(t_cap, CO2_captured_3d, linewidth=2, color="#9467bd")

style_ax(ax, "Total CO$_2$ Captured vs Time",
         "Time  [hr]", "CO$_2$ Captured  [mol]")
plt.tight_layout()
plt.show()
# ── 3. Sorbent loading vs time ────────────────────────────────────────────────
time_axis       = np.linspace(0, t_f, len(qb_hist))
loading_vs_time = [np.mean(q) * ms_pp * 2 for q in qb_hist]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_axis / 3600, loading_vs_time, linewidth=2, color="#2ca02c")

style_ax(ax, "Spatially Averaged Sorbent Loading vs Time",
         "Time  [hr]", "Loading  [mol m$^{-2}$]")
plt.tight_layout()
plt.show()

# ── 4. Total CO₂ captured ─────────────────────────────────────────────────────
W = 1.0
CO2_captured_3d = [val * W for val in CO2_captured]

fig, ax = plt.subplots(figsize=(8, 4.5))
t_cap = t_array[:len(CO2_captured_3d)] / 3600

ax.plot(t_cap, CO2_captured_3d, linewidth=2, color="#9467bd")

style_ax(ax, "Total CO$_2$ Captured vs Time",
         "Time  [hr]", "CO$_2$ Captured  [mol]")
plt.tight_layout()
plt.show()