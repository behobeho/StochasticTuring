import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

steps = 5000 #number of time steps per trajectory
length = 200 #size of the grid

#initialise reaction parameters
#parameter values have been determined using the PRE-DIFFUSION 2-node Turing Gillespie Script
k1 = 0.50
k2 = 1.00
k3 = 0.51
k4 = 1.00
dA = 10  #diffusion constant A
dI = 1000  #diffusion constant I
pA = 100   #production of A
pI = 1  #production of I
muA = 1  #degradation of A
muI = 10  #degradation of I

#initial conditions
A0 = 10
I0 = 10

#creating the grid
time = np.zeros(steps+1)
concA = np.zeros((length, steps+1))
concI = np.zeros((length, steps+1))

#set initial conditions
concA[:, 0] = A0
concI[:, 0] = I0

for step in range(1, steps+1):

    #update concentrations
    concA[:, step] = concA[:, step-1] 
    concI[:, step] = concI[:, step-1]

    for box in range(1, length): #iterate through each compartment in the timestep
        rT = k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA + muI + dA*concA[box,step] + dI*concI[box,step]#rate total, normalisation factor
        Rand1 = np.random.uniform(0,1) #random number for reaction selection
        Rand2 = np.random.uniform(0,1) #random number for time increment
        
        if 0 < Rand1 <= (k1*concA[box,step]) / rT: #self activation
            concA[box, step] += 1

        elif (k1*concA[box,step]) / rT < Rand1 <= (k1*concA[box,step] + k2*concI[box, step]*concA[box, step]) / rT and concA[box, step]>0: #inhibition
            concA[box, step] -= 1

        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step]) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step]) / rT: #activation
            concI[box, step] += 1

        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step]) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step]) / rT and concI[box, step]>0: #self inhibition
            concI[box, step] -= 1

        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step]) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA) / rT: #basal production of activator
            concA[box, step] += 1
        
        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA +pI) / rT: #basal production of inhibitor
            concI[box, step] += 1

        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA) / rT and concA[box, step]>0: #basal production of activator
            concA[box, step] -= 1
        
        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA + muI) / rT and concI[box, step]>0: #basal production of inhibitor
            concI[box, step] -= 1
        
        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA + muI) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA + muI + dA*concA[box,step]) / rT and concA[box, step]>0: #Diffusion of A
            pLR = np.random.uniform(0,1) #probability of moving left or right
            
            if box == length-1: #molecules in the far-right compartment have to diffuse left
                concA[box, step] -= 1
                concA[box-1, step] += 1

            elif box == 0: #molecules in the far-left compartment have to diffuse right
                concA[box, step] -= 1
                concA[box+1, step] += 1

            elif 0 <= pLR < 0.5 and box != 0: #diffuse to the left with probability p=1/2
                concA[box, step] -= 1
                concA[box-1, step] += 1

            elif 0.5 <= pLR < 1 and box != length-1: #diffuse to the right with probability p=1/2
                concA[box, step] -= 1
                concA[box+1, step] += 1
        
        elif (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA) / rT < Rand1 <= (k1*concA[box, step] + k2*concI[box, step]*concA[box, step] + k3*concA[box, step]*concI[box, step] + k4*concI[box, step] + pA + pI + muA + muI) / rT and concI[box, step]>0: #basal production of inhibitor
            pLR = np.random.uniform(0,1)
            if box == length-1: #molecules in the far-right compartment have to diffuse left
                concA[box, step] -= 1
                concA[box-1, step] += 1

            elif box == 0: #molecules in the far-left compartment have to diffuse right
                concA[box, step] -= 1
                concA[box+1, step] += 1

            if 0 <= pLR < 0.5 and box != 0: #diffuse to the left with probability p=1/2
                concI[box, step] -= 1
                concI[box-1, step] += 1

            else: #diffuse to the right with probability p=1/2
                concI[box, step] -= 1
                concI[box+1, step] += 1

    time[step] = time[step-1] - np.log(Rand2)/rT

#plot heatmap of activator and inhibitor    
sns.heatmap(concA)
plt.title("Activator Copy Number With Diffusion")
plt.xlabel("Time (s)")
plt.ylabel("Compartment")
plt.show()

sns.heatmap(concI, cmap='mako')
plt.title("Inhibitor Copy Number With Diffusion")
plt.xlabel("Time (s)")
plt.ylabel("Compartment")