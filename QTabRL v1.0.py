# -*- coding: utf-8 -*-
"""

Software Agents Coursework - Q-Learning on a (Slippery) Frozen Lake
Version 1.0
Original Code Base -
A. Juliani, 2016. Matrix Q Learning Code Base Code. 
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-
0-q-learning-with-tables-and-neural-networks-d195264329d0
Revision perfomed by - mike.a.taylor
"""

##############################################################################
# Library Imports
##############################################################################

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

##############################################################################
# Functions
##############################################################################

def actionSelect(s,Q,e,env):
    random = np.random.randn(1,env.action_space.n)
    a = np.argmax(Q[s,:] + e*random)
    return a

def printValues(Q):
    V = np.max(Q, axis=1)
    print (np.reshape(V, [16, 4]))

def printPolicy(Q):
    maxV = np.argmax(Q, axis=1)
    maxV = np.reshape(maxV, [16, 4]).astype('str')
    maxV[maxV == '0'] = 'L'
    maxV[maxV == '1'] = 'D'
    maxV[maxV == '2'] = 'R'
    maxV[maxV == '3'] = 'U'
    print (maxV)
  
def Qlearning(alpha,lrdecay,y,epsilon,penaliseFall,penaliseSlow,penLength,dr,num_episodes,trueMove,maxMoves,window):
    #create lists to contain total rewards and steps per episode  +constants for graphical output, averaging etc.
    rList = []
    jList = []
    sList = []
    rWindowList = []
    window = window
    convergence = 0
    
    print("")
    print("Specific Q Learning Parameters")
    print("y =", y)
    print("alpha =", alpha)
    print("alpha decay =", lrdecay)
    print("epsilon decay rate =", dr)
    e = epsilon
    lr = alpha
    #Initialize table with all zeros
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    for i in range(num_episodes):
        #Logging for visualisations
        rLen = min(len(rList),window)
        rTmp = sum(rList[-rLen:])/(rLen+1)
        rWindowList.append(rTmp)
        #Reset environment and get first new observation
        s = env.reset()
        e = e*dr
        lr = lr*lrdecay
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < maxMoves:
            j+=1
            #Choose an action by greedily (w/ noise) picking from Q table... 0-left; 1-down; 2-right; 3-up
            a = actionSelect(s,Q,e,env)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            if trueMove == True:            
                if s-1 == s1: 
                    a = 0 #left
                elif s +8 == s1:
                    a = 1 #down correct
                elif s + 1 == s1:
                    a = 2#leftcorrect
                elif s - 8 == s1: 
                    a = 3 # right
            if penaliseFall == True:
                if s1 in thinIce:
                    r = -0.1
            #penalise slow models
            if penaliseSlow == True:
                if j>penLength:
                    r = r/(np.log10(1+j))
            #Update Q-Table with new knowledge
            knowledge = lr*(r + y*np.max(Q[s1,:])-Q[s,a] )
            Q[s,a] = Q[s,a] + knowledge # i.e. old value, + learning rate(reward + discount factor y * optimal future value estimate - old Q value)
            if knowledge <0.001:
                convergence += 1
            else:
                convergence = 0
            rAll += r
            s = s1
            if d == True:
                break                   
        #if convergence >1000:
         #   print ("Converged @", i)
        jList.append(j) 
        rList.append(rAll)
        sList.append(s)
        
    df1 = pd.DataFrame({'lr':lr,'y':y,'edr':dr,'j': jList, 'r' : rList, 's':sList, "rwin":rWindowList})
    print("")
    print("Q Learning Results...")
    print ("Percent of succesful episodes: " + str(100*sum(rList)/num_episodes) + "%")
    print ("Average Score over time: " +  str(sum(rList)/num_episodes))
    return(df1,Q,rList,jList,sList,rWindowList)

##############################################################################
# Main Code
##############################################################################

# Create Environment
env = gym.make('FrozenLake8x8-v0')
print (env.render())

# global Q learning parameters
num_episodes = 1000
maxMoves = 200
window = 100
repeats = 1
penaliseFall = True
trueMove = False
penaliseSlow = False
penLength = 1
thinIce = [19,29,35,41,42,46,49,52,54,59]

# Set alpha (learning rate) parameters
alphaIni = .9 
alphaFin = 1.0 
alphaInc = .05
lrdecay = 1.0#0.99995

# Set y parameters
yIni = .99 
yFin = 1.0 
yInc = .05 

# Set epsilon decay parameters
epsilon = 1.0
drIni = 0.995 
drFin = 0.996  
drInc = 0.001 

#create empty df
columns = ['lr','y','edr','j','r','s']
dfFinal = pd.DataFrame(columns=columns)

#create hyperparameter ranges
alphaRange = np.arange(alphaIni,alphaFin,alphaInc)
yRange = np.arange(yIni,yFin,yInc)
drRange = np.arange(drIni,drFin,drInc)

# overwritten, to simplify selection, grid search also viable
alphaRange = [0.8] 
yRange = [0.9]
drRange = [0.995]

print("Global Q Learning Parameters")
print("Maximum Number of Moves =", maxMoves)
print("Episodes =", num_episodes)
print("Repeats = ", repeats)
print("penalise Fall = ", penaliseFall)
print("trueMove = ", trueMove)
print("penalise Slow = ", penaliseSlow)

#grid search across learning parameters
for alpha in alphaRange:
    for y in yRange:
        for dr in drRange:
            for repeat in range(repeats):
                df1,Q,rList,jList,sList,rWindowList = Qlearning(alpha,lrdecay,y,epsilon,penaliseFall,penaliseSlow,penLength,dr,num_episodes,trueMove,maxMoves,window) 
                dfFinal = dfFinal.append(df1)

settings= str((alpha,lrdecay,y,epsilon,penaliseSlow,penLength,dr,num_episodes,trueMove,maxMoves,window))
#Final csv storage
dfFinal.to_csv("Results" + settings +".csv", mode='a', header=False)

##############################################################################
# Results
##############################################################################

timestr = time.strftime("%Y%m%d-%H%M%S")

np.set_printoptions(precision=3, suppress=True)
print ("Q function:")
print (Q)

print ("Value function:")
printValues(Q)

print ("Policy:")
printPolicy(Q)

print("")
print("Q Learning Average Rewards Over Time")
plt.ylim(0, 1.0)
plt.xlim(0, num_episodes)
plt.plot(rWindowList)
plt.ylabel('Prev.' + str(window) + ' Episode Average Reward')
plt.xlabel('Episode')
plt.savefig("Line" + settings + timestr +'.png')
plt.show()
jlistsum = 0

print("")
print("Q Learning Episode Length Scatter")
plt.ylim(0, maxMoves)
plt.xlim(0, num_episodes)
for i in range(num_episodes):
    if sList[i] == 63:
        plt.scatter(i,jList[i],color='b',alpha=0.2,lw=0)
        jlistsum += jList[i]
    else:
        plt.scatter(i,jList[i],color='r',alpha=0.2,lw=0)
plt.ylabel('Path Length')
plt.xlabel('Episode')
plt.savefig("Scatter" + settings + timestr +'.png')
plt.show()

resultsWindow = 6000
resultsStart = num_episodes - resultsWindow
successCount = sum(x is 63 for x in sList[resultsStart:])
thinIceCount = sum(x in thinIce for x in sList[resultsStart:])
print("Success Count - ", successCount)
print("Success Percent - ", 100*successCount/resultsWindow)
print("Thin Ice Count - ", thinIceCount)
print("Thin Ice Percent - ", 100*thinIceCount/resultsWindow)
print("Time Out Count - ", (resultsWindow - thinIceCount - successCount))
print("Time Out PErcent - ", 100*(resultsWindow - thinIceCount - successCount)/resultsWindow)
print("Not failed count- ", (resultsWindow - thinIceCount))
print("Not failed Percent - ", 100*(resultsWindow - thinIceCount)/resultsWindow)
print("Average Run Length - ", sum(jList[resultsStart:])/resultsWindow)
print("Average Success Run Length - ", (jlistsum/successCount))
