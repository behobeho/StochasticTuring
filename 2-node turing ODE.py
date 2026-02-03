import numpy as np
import matplotlib.pyplot as plt

steps = 1000 #number of time steps per trajectory
length = 100 #size of the grid

#initialise reaction parameters
k1 = 0.50
k2 = 1.00
k3 = 0.51
k4 = 1.00
dA = 0.025  #diffusion constant A
dI = 0.075  #diffusion constant I
pA = 0.06   #production of A
pI = 0.06   #production of I
muA = 0.05  #degradation of A
muI = 0.05  #degradation of I

#variables
#dI = np.linspace(0,0.1,101)
#R1 = np.random.uniform(0,1,steps+1)
#R2 = np.random.uniform(0,1,steps+1)

#initial conditions
A0 = 3
I0 = 3

#creating the grid
t = np.zeros(steps+1)
rxn = np.zeros((length, 2, steps+1)) #3D array containing concentrations of A and I in each compartment at each timestep


###  PART 1: Checking for Pre-Diffusion Stability ###

#set initial conditions
rxn[:,0,0] = A0
rxn[:,1,0] = I0

for t in range(steps+1):
    for box in range(length):
        rT = k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA + pI + muA + muI #rate total, normalisation factor
        Rand1 = np.random.uniform(0,1)
        Rand2 = np.random.uniform(0,1)

        if 0 < Rand1 <= k1*rxn[box,0,t] / rT: #self activation
            rxn[box,0,t] += 1

        elif k1*rxn[box,0,t] / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t]) / rT and rxn[box,0,t]>0: #inhibition
            rxn[box,0,t] -= 1

        elif (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t]) / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t]) / rT: #activation
            rxn[box,1,t] += 1

        elif (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t]) / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t]) / rT and rxn[box,1,t]>0: #self inhibition
            rxn[box,1,t] -= 1

        elif (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t]) / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA) / rT: #basal production of activator
            rxn[box,0,t] += 1
        
        elif (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA) / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA +pI) / rT: #basal production of activator
            rxn[box,1,t] += 1

        elif (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA + pI) / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA + pI + muA) / rT and rxn[box,0,t]>0: #basal production of activator
            rxn[box,0,t] -= 1
        
        elif (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA + pI + muA) / rT < Rand1 <= (k1*rxn[box,0,t] + k2*rxn[box,1,t]*rxn[box,0,t] + k3*rxn[box,0,t]*rxn[box,1,t] + k4*rxn[box,1,t] + pA + pI + muA + muI) / rT and rxn[box,1,t]>0: #basal production of inhibitor
            rxn[box,1,t] -= 1

print(rxn)
