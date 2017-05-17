# -*- coding: utf-8 -*-
"""
Title - Software Agents Coursework - Q-Learning on a (Slippery) Frozen Lake
Version - 1.0
Description - A Q table based implmentation to identify optimal strategies to traverse the frozen lake
Original Code Base - A. Juliani, 2016. Matrix Q Learning Code Base Code. 
Orifinal Code Link - https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
Revisions/Updates - perfomed by - mike.a.taylor

Execution Instructions
1. Import Gym library as required (available via PIP)
2. Import Tensorflow as required (available via PIP)
3. Select paramters (lines 37 to 58)
4. Execute Code

Additional (Planned) Improvements 
1. Port Neural Network to a callable function
2. Add in a grid search for the learning
3. Add in heatmap trails - success Blue, failure red
4. Add in a reward for the speed of the path vs the safe path + negative rewward for falling in hole
5. Add in a replay/reset if after x tries you're still failing (caught in local minima, tie decay to success?/kick in decay after success?)

"""

##############################################################################
# Library Imports
##############################################################################

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##############################################################################
# Main Code
##############################################################################

# Set learning parameters
env = gym.make('FrozenLake8x8-v0')
n = 64 # number of possible states
possMoves = 4 # number of actions/moves possible
lr = 0.9 # alpha
y = .99 # gamma
e = 1.0 # epsilon
edecay = 0.9995
rewardAdjust = False # explicitly reward a quick journey over the ice
trueMove = True
maxMoves = 200
num_episodes = 1000
GPU = False
penalty = False
thinIce = [19,29,35,41,42,46,49,52,54,59]

#create lists to contain total rewards and steps per episode (also windows to (crudely) smooth the output graphic)
jList = []
rList = []
rWindowList = []
window = 100
rWinMax = 0

print("")
print("Q Learning Parameters")
print("y =", y)
print("e =", e)
print("alpha =", lr)
print("Maximum Number of Moves =", maxMoves)
print("Episodes =", num_episodes)
print("GPU used?", GPU)


#These lines establish the feed-forward part of the network used to choose actions
if  GPU == True:
    config = tf.ConfigProto(device_count = {'GPU': 0}) # toggles GPU usage
tf.reset_default_graph()
inputs1 = tf.placeholder(shape=[1,n],dtype=tf.float32) # tf placeholder, instantiates the input shape to the tf ANN (the flattened state matrix)
W = tf.Variable(tf.random_uniform([n,possMoves],0,0.01)) # initialise the weights within the tf ANN (shape, mincval,maxval)
Qout = tf.matmul(inputs1,W) # Qout predicts what it thinks the Q will be for the 4 moves, it's the multiplication of the input matrix and the ANN weight matrix (which is sort of eq. to the Q matrix)
predict = tf.argmax(Qout,1) # choose the best mvoe from Qout

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,possMoves],dtype=tf.float32) #  tf placeholder, instantiates the input shape to the tf ANN
loss = tf.reduce_sum(tf.square(nextQ - Qout)) # sum of squared errors (i.e. Qout what I thought I'd get, nextQ what I actually get)
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr) # training model object, with 0.1 learning rate
updateModel = trainer.minimize(loss) # trains (well updates) with the trainer object instantiated above

#Initialise the neural network model
init = tf.global_variables_initializer()

#Main Q leanring 
with tf.Session() as sess: #config=config
    sess.run(init)
    
    #Monitoring/logging
    for i in range(num_episodes):
        rLen = min(len(rList),window)
        rTmp = sum(rList[-rLen:])/(rLen+1)
        rWindowList.append(rTmp)
        if rTmp > rWinMax:
            rAverageMax = rTmp
            #print ("Best 100 episode average reward - ", r100Max, ", Episode", i)
        if rTmp >= 0.99: 
            print ("Solved!!!!")
            break
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        j = 0
        #The Q-Network
        while j < maxMoves:
            e = e * edecay
            j+=1
            if rewardAdjust == True:
                rewardWeight = 1/np.log10(j)
            else:
                rewardWeight = 1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(n)[s:s+1]}) # tensor sess does predict -> a (choose an action), Qout -> AllQ (refresh the Q values, bit out of place, more naturl at end, but immaterial); "training data" = tfplaceholder inputs, i.e. the current state
            if np.random.rand(1) < e: #the epsilon random component, if so the randomise the step
                a[0] = env.action_space.sample() #a[0] as sess run outputs an array, otherwise irrelevant, env.action... is jsut a random action decider
            #Take a step... get new state and reward from environment
            s1,r,d,_ = env.step(a[0]) # state, reward, episode end, "_" is irrelevant, it's for diagnostics
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(n)[s1:s1+1]}) #Qout given new state, (ie feed_dict..)
            if trueMove == True:            
                if s-1 == s1:
                    a[0] = 0
                elif s +8 == s1:
                    a[0] = 1
                elif s + 1 == s1:
                    a[0] = 2
                else:
                    a[0] = 3
            if penalty == True:
                if s1 in thinIce:
                    r = -0.1
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1) # best next move
            targetQ = allQ # refreshes the Q values in the old matrix, ie at current state
            targetQ[0,a[0]] = r + y*maxQ1 # targetQ updated at action (u/d/l/r), i.e. actual reward(r) + gamma*the best Q value(u/d/l/r)

            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(n)[s:s+1],nextQ:targetQ})# Updates the matrix weights given the experience NB "_" is irrelevant, it's for diagnostics
            #sess.run([updateModel,W],feed_dict={inputs1:np.identity(n)[s:s+1],nextQ:targetQ})# Updates the matrix weights given the experience NB "_" is irrelevant, it's for diagnostics

            # Log the reward
            rAll += r * rewardWeight #add to the overall reward 
            s = s1 # set new state
            if d == True:
                break            
        jList.append(j)
        rList.append(rAll)

print("")
print("Q Learning Results...")
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
print ("Average Score over time: " +  str(sum(rList)/num_episodes))

print("")
print("Q Learning Results Over Time...")
plt.plot(rWindowList)
plt.ylabel('Prev.' + str(window) + ' Episode Average Reward')
plt.xlabel('Episode')
plt.show()

