import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

"""
Authors: Jaxon Cione, Chuan Heng, and Andrew Hickman
Date: 2/28/2026
Description: This code solves the PDE for a 2D adsorption process using finite volume/ Tridiagonal matrix method. Gas phase is treated as
psuedo-steady at each timestep. Which we found to be a reasonable assumption after talking with our fluid dynamics professor and from "solving" the semi-analytic solution
"""

def black_box(T, H, t_f):
    #parameters------------------------------------------------------------------
    mw_CO2=44.01/1000 #kg/mol, molecular weight of CO2
    ms_pp=1 #kg/m^2 of sorbent
    rho=1000 #kg/m^3, density of sorbent. We don't know this but we can change it later
    R=8.314 #J/(mol*K)
    C_0=.017 #mol/m^3... ambient CO2 conecentration .01725
    viscosity=1.81E-5 #of air at room temp, kg/m/s
    D_A=1.6E-5 #diffusion coeff m^2/s
    L=1 #m length of the bed

    #values given from Neeraj ------------------------------------------------------------------
    tau=.4
    k_0=2000 #1/s
    Xi=.7
    D_H=70000 #J/mol delta H of adsorption
    E_a=40000 #J/mol #activation energy for adsorption
    T_0=298.15 #K 
    q_m0=3 #mol-CO2/kg
    b_0=2E-14 #1/Pa

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
    # print("alpha =", alpha)

    #Solving right hand side via finite volume method------------------------------------------------------------
    #right had side of the ODE system, will be used in the ODE solver qb is bottom loading qt is top loading.
    def C_ss(qb):
        C = np.zeros((N_x, N_y))
        beta_j = (D_A * dx) / (v_profile * dy**2)

        #inlet column
        rhs = np.full(N_y, C_0)
        banded = np.zeros((3, N_y))

        # Interior rows
        banded[1, :] = 1 + 2*beta_j
        banded[0, 1:] = -beta_j[:-1]    
        banded[2, :-1] = -beta_j[1:]     

        #boundary conditiosn
        gamma_b = (k * ms_pp * dy / D_A) * (q_e_vec(C_0) - qb[0])
        banded[1, 0] = 1 + 2*beta_j[0]  
        banded[0, 1] = -2*beta_j[0]     
        rhs[0] -= 2*beta_j[0] * gamma_b

        #top bc is the same due to sym
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

            #top usiny symmetry
            banded_matrix[1, -1]=1+beta_j[-1]

            C[i] = solve_banded((1, 1),banded_matrix, rhs)

        return np.clip(C, 0, C_0) #clipping s.t we dont have unphysical values



    dt=5#time step in seconds
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
        # C_adsorbed_current = np.sum(q_total_current *ms_pp*(dx))
        C_adsorbed_current = np.sum(qb) * ms_pp * dx * 1 * 2     
        CO2_captured.append(C_adsorbed_current)
        C_outlet_avg.append(C[-1, :].mean()) #grabbing the mean for plotting

    C_snaps = {}
    C = C_ss(qb) #solving one last time
    C_snaps[36000]= C.copy() #storing a copy of all 10 hours

    q_total=np.sum(qb)*2*dx*1#total loading from top and bottom hence multiplication by 2 #assuming symettry 
      #mol/m^2 will be comparing with comsol

    ### DEFINING THE BLACK BOX 
    
    total_CO2_captured = q_total # total CO2 captured per cycle
    
    N_cycle = 2.6298e6 / (2 * t_f)   # cycles per month

    deltaP = alpha * 2 * viscosity * L
    
    # energy cost per cycle
    
    c = 0.5 # user input depending on prefrence on number of cycles
    
    E_cycle = c * N_cycle
    
    eta = 0.5
    E_ads = v_avg_target * -deltaP / eta * t_f / 1000 * L * H   # kJ/cycle
    E_des = 50 * total_CO2_captured                            # kJ/cycle
    E_tot = E_ads + E_des +E_cycle

    e_grid = 100       # kg CO2 / GJ
    CO2_emission = e_grid * E_tot / 1e6   # kg CO2 / cycle
    
    total_CO2_captured_Kg = total_CO2_captured * mw_CO2

    return N_cycle * (total_CO2_captured_Kg - CO2_emission)


T = 286.01
H = 0.010593
t_f = 7093.5203

T = 286.0967
H = 0.013342
t_f = 6914.6773

T = 286.0794
H = 0.013048
t_f = 7011.3000

T = 286.0171
H = 0.012329
t_f = 6997.6989

T = 286.0604
H = 0.010884
t_f = 6719.3420

T = 286.4494
H = 0.010000
t_f = 7000.0453

print(black_box(T, H, t_f))
