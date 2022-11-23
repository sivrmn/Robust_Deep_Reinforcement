# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:23:23 2017

@author: rajag038

Adding random gaussian noise to all the states while training to simulate the 
effect of variations in system parameters. 
This random noise is added randomly some of the time. 
"""

#==============================================================================
# Imports
#==============================================================================
import random
import numpy as np
import matplotlib.pyplot as plt
import time as time


import gym

from collections import deque
import json
#import h5py
from keras.models import Sequential
from keras.models import model_from_json
#from keras.models import h5py
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
#import tensorflow as tf
import os

#==============================================================================


#==============================================================================
# Program Constants
#==============================================================================

OBSERVATION = 10000 # Timesteps to observe before training

GAMMA = 0.99 #0.99 # Decay rate of past observations

#-- Exploration - Explotiation balance --#
EXPLORE = 200000 # Frames over which to anneal epsilon
FINAL_EPSILON = 0.05#0.05 # Final value of epsilon
INITIAL_EPSILON = 0.2#0.8 # Starting value of epsilon

#-- Training parameters --#
TRAIN_INTERVAL = 1
REPLAY_MEMORY = 200000 # Number of previous transitions to remember
BATCH = 32 # Size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4


#-- Reward selection --#
REWARD_LOSS = -1
REWARD_NOLOSS = 0.1
#==============================================================================


#==============================================================================
# Building Q-Function model structure
#==============================================================================
def build_model():
    #print("Now we build the model")
    model = Sequential()
    model.add(Dense(4, input_dim = 4, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(2, activation = 'relu'))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)    
    #model.compile(loss = 'mse', optimizer = 'sgd')
    #print("We finish building the model")
    return model    
#==============================================================================



#==============================================================================
# Training the network
#==============================================================================
def train_network(model, env, init_s, modelName):

    #-- Program Constants --#
    ACTIONS = env.action_space.n # Number of valid actions 
    REND = 0
    RECORD_DIV = 1000
    #-----------------------------------------------------#

    #-- Variable initializations --#
    theta_threshold_radians = 120 * 2 * np.pi / 360
    x_threshold = 2.4       
    
    done = False
    t = 0 
    lclT = 0
    survT = 0
    survCnt = 0
    r_t = 0
    a_t = 0

    loss = 0
    Q_sa = 0     
    rcdCnt = 0 
    Q_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    Loss_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    survT_Arr = np.zeros(np.int(EXPLORE/RECORD_DIV))
    
    s_t = init_s
    s_t = np.array(s_t.reshape(1, s_t.shape[0]))
    
    stateRecords = np.matrix(np.matrix(s_t))
    
    #-- Storage for replay memory --#
    D = deque()
    

    #-- Exploration of the game state space begins --#
    epsilon = INITIAL_EPSILON
    
    
    
    start_time = time.time()
    
    while t < EXPLORE:
                       
        if done:        
            if(lclT>=5):
                s_t = env.reset()
                s_t = np.array(s_t.reshape(1, s_t.shape[0]))
                lclT = 0
                survCnt = 0
                done = 0
                
            lclT = lclT + 1

        if(done == 1):
            survT = survCnt            
        else:
            survCnt = survCnt + 1            
            
            
        #-- Choosing an epsilong greedy action --#
        if np.random.random() <= epsilon:
            a_t = env.action_space.sample()
        else:
            q = model.predict(s_t)
            max_Q = np.argmax(q)
            a_t = max_Q

        
        #-- Exploration annealing --#
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            

        #observation, reward, done, info = env.step(action)
        [s_t1, r_t, done, info] = env.step(a_t)
        
        # Adding Gaussian Noise to States
        if(t < OBSERVATION): 
            stateRecords = np.append(stateRecords,np.matrix(s_t1), 0)
        elif(t == OBSERVATION):
            s_t_std = np.std(stateRecords,0)
            s_t_std = np.array(s_t_std)
            s_t_std = s_t_std.flatten() 
        else:
            if(np.random.rand(1)>1):
                s_t_noise = 0#(0.02*np.random.randn(np.size(s_t)) + 1)*s_t1 #s_t_std
            else:
                s_t_noise = 0
            s_t1 = s_t1 + s_t_noise
            
        
        
        # Creating reward function
     
        if( (abs(s_t1[0]) < x_threshold) & (abs(s_t1[2]) < theta_threshold_radians) ):
            r_t = REWARD_NOLOSS
        else:
            r_t = REWARD_LOSS
            
            
       
        s_t1 = np.array(s_t1.reshape(1, s_t1.shape[0]))

        D.append([s_t, a_t, r_t, s_t1, done])

        
        #-- Update graphics based on action taken --#
        if(REND == 1):
            env.render()
                

        
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        
        #-- Training after the initial observation is complete --#
        if((t > OBSERVATION)  & (t % TRAIN_INTERVAL == 0)):


            minibatch = random.sample(D, BATCH) 
            
            inputs = np.zeros((BATCH, s_t1.shape[1]))

            targets = np.zeros((inputs.shape[0], ACTIONS))

            for batchCnt in range(0, len(minibatch)):
                state_t = minibatch[batchCnt][0]
                action_t = minibatch[batchCnt][1]
                reward_t = minibatch[batchCnt][2]
                state_t1 = minibatch[batchCnt][3]
                terminal = minibatch[batchCnt][4]
    
                inputs[batchCnt:batchCnt+1] = state_t
                targets[batchCnt] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
    
                #-- Bellman-Deep Q update equations --#
                if terminal:
                    targets[batchCnt, action_t] = reward_t
                else:
                    targets[batchCnt, action_t] = reward_t + GAMMA * np.max(Q_sa)
    
                    
            loss = model.train_on_batch(inputs, targets)


        s_t = s_t1
        t = t + 1

        #-- Saving progress every 1000 iterations --#
        if t % RECORD_DIV == 0:
            print('Saving Model')
            model.save_weights(modelName+".h5", overwrite = True)
            with open(modelName+".json", "w") as outfile:
                json.dump(model.to_json(), outfile)
                
            # Local heuristic to stop if sufficiently high 'Q' is reached
            #if(survT>=499):
                #t = EXPLORE
   
            
        #-- Print updates of progress every 1000 iterations --#
        if t % RECORD_DIV == 0:
            Q_Arr[rcdCnt] = np.max(Q_sa)
            Loss_Arr[rcdCnt] = loss
            survT_Arr[rcdCnt] = survT
            rcdCnt = rcdCnt + 1
            
            print("TIMESTEP", t, "SURVIVAL TIME",survT ,"/ EPSILON", np.round(epsilon,3), "/ ACTION", a_t, "/ REWARD", r_t,  "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
                    
    end_time = time.time()
    print('Execution time')
    print(end_time - start_time)
    print('Time per iteration')
    print((end_time - start_time)/EXPLORE)                
                    
                
    return(Q_Arr, Loss_Arr, survT_Arr)
#==============================================================================


#==============================================================================
# Loading a trained model
#==============================================================================
def load_model(model, file_name):
    model.load_weights(file_name)
    #model.compile(loss = 'mse', optimizer = 'sgd')
    return model    
#==============================================================================



#==============================================================================
# Select between model training and evaluation
#==============================================================================
def deepQ(select, game, modelName, parameter):
    avgT = 0
    Q_Arr = 0
    Loss_Arr = 0
    survT_Arr = 0
    if(select == 'Train'):
        model = build_model()
        #model = load_model(modelTemp, 'model2.h5') # Loading previously trained model
        env = gym.make(game)
        #-- Updating parameters --#        
        env.env.gravity = parameter[0]
        env.env.masscart = parameter[1]
        env.env.masspole = parameter[2]
        env.env.length = parameter[3]  # actually half the pole's length
        env.env.force_mag = parameter[4]
        env.env.tau = parameter[5]
        
        init_s = env.reset()
        [Q_Arr, Loss_Arr, survT_Arr] = train_network(model, env, init_s, modelName)
        
        plt.figure()
        plt.plot(Q_Arr)
        plt.ylabel('Q-Value')
        plt.xlabel('Samples (x1000)')
        plt.title('Q-Values During Training')
        plt.savefig('Q_Values_'+modelName+'.eps', format='eps', dpi=1000)
        plt.savefig('Q_Values_'+modelName+'.jpeg', format='jpeg', dpi=1000)
        
        plt.figure()
        plt.plot(Loss_Arr)
        plt.ylabel('Loss Function')
        plt.xlabel('Samples (x1000)')
        plt.title('Neural Network Loss Function')
        plt.savefig('Loss_'+modelName+'.eps', format='eps', dpi=1000)
        plt.savefig('Loss_'+modelName+'.jpeg', format='jpeg', dpi=1000)

        
        plt.figure()
        plt.plot(survT_Arr)
        plt.ylabel('Survival Time (s)')
        plt.xlabel('Samples (x1000)')
        plt.title('Survival Time During Training')
        plt.savefig('SurvivalTime_'+modelName+'.eps', format='eps', dpi=1000)
        plt.savefig('SurvivalTime_'+modelName+'.jpeg', format='jpeg', dpi=1000)        
        
    elif(select == 'Test'):
                
        model = build_model()
        env = gym.make(game)
        
        #-- Updating parameters --#
        env.env.gravity = parameter[0]
        env.env.masscart = parameter[1]
        env.env.masspole = parameter[2]
        env.env.length = parameter[3]  # actually half the pole's length
        env.env.force_mag = parameter[4]
        env.env.tau = parameter[5]
        
        file_name = modelName+".h5"
        load_model(model, file_name)
                                    
        #-- Evaluation --#
        avgT = np.zeros(10)
        avgTR = np.zeros(10)
        for i_episode in range(10):
            observation = env.reset()
            t = 0
            done = 0
            while(done == 0):
                #env.render()
                ## play the game with model
                s_t = observation.reshape(1, observation.shape[0])
                q = model.predict(s_t)
                action = np.argmax(q)
                observation, reward, done, info = env.step(action)
                t = t + 1
                if done:
                    #print("Survival time =  {} timesteps".format(t))
                    avgT[i_episode] = t

        '''
        for i_episode in range(10):
            observation = env.reset()
            t = 0
            done = 0
            while(done == 0):
                env.render()
        
                ## play the game randomly
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                t = t + 1
                if done:
                    print("Survival time =  {} timesteps".format(t))
                    avgTR[i_episode] = t
        '''
        print("\n")
        #print("Average Peformances")
        #print("Average RL survival time = {} timesteps".format(np.mean(avgT)))   
        #print("Average Random survival time = {} timesteps".format(np.mean(avgTR)))
        #print("Percentage improvement = {}".format((np.mean(avgT)*100)/(np.mean(avgTR))))
    return(Q_Arr, Loss_Arr, survT_Arr, np.mean(avgT))
#==============================================================================


#==============================================================================
# Main function area
#==============================================================================
#[Q_Arr, Loss_Arr, survT_Arr] = deepQ('Test', 'CartPole-v1', 'modelrob4')
#==============================================================================


import scipy as sc

#==============================================================================
# Pessimiser
#==============================================================================
parameter=np.zeros(4)
parameter[0]=9.8 # gravity
parameter[1]=1.0 # masscart
parameter[2]=0.1 # masspole
parameter[3]=0.5 # length
#parameter[4]=10.0 # force_mag
#parameter[5]=0.02 # tau
#==============================================================================

#==============================================================================
# 
#==============================================================================
class Pessimiser:
    #==============================================================================
    #     
    #==============================================================================
    def __init__(self):
    
        self.cnt  = 1

    #==============================================================================
    
        
    #==============================================================================
    # Objective function to minimize
    #==============================================================================
    def getSurvT(self, parameter):
            
        newParameter=np.zeros(6)
        newParameter[0]= 9.8*parameter[0] # gravity
        newParameter[1]= 1.0*parameter[1]# masscart
        newParameter[2]= 0.1 *parameter[2] # masspole
        newParameter[3]= 0.5*parameter[3] # length
        newParameter[4]= 10 # force_mag
        newParameter[5]= 0.02 # tau
        [Q_Arr, Loss_Arr, survT_Arr, survT] = deepQ('Test', 'CartPole-v1', 'model'+str(self.cnt), newParameter)
        #[Q_Arr, Loss_Arr, survT_Arr, survT] = deepQ('Test', 'CartPole-v1', 'modelrob4', newParameter)
        
        print(parameter)
        print(survT)
        return(survT)
    #==============================================================================
    
    
    #==============================================================================
    # Objective function to minimize
    #==============================================================================
    def trainNew(self, parameter):
            
        newParameter=np.zeros(6)
        newParameter[0]= 9.8*parameter[0] # gravity
        newParameter[1]= 1.0*parameter[1]# masscart
        newParameter[2]= 0.1 *parameter[2] # masspole
        newParameter[3]= 0.5*parameter[3] # length
        newParameter[4]= 10 # force_magd
        newParameter[5]= 0.02 # tau
        [Q_Arr, Loss_Arr, survT_Arr, survT] = deepQ('Train', 'CartPole-v1', 'model'+str(self.cnt), newParameter)
        
        print(parameter)
        print(survT)
        return(survT)
    #==============================================================================

    
#==============================================================================

#%%

pesObj = Pessimiser()
pesObj.cnt = 1
for ii in range(0,1):
    lBnd = 0.1
    uBnd = 2.0
    x0 = np.ones(np.shape(parameter))*lBnd 
    
    # Computing bounds
    xmin = np.ones(np.shape(parameter))*lBnd
    xmax = np.ones(np.shape(parameter))*uBnd
    
    bnds = [(low, high) for low, high in zip(xmin, xmax)]
    
    
    # Simple optimization
    #res = sc.optimize.minimize(getSurvT, x0, method='Nelder-Mead',options={'disp': True, 'maxiter' : 100}, tol = 10, bounds=bnds)
    
    
    
    #==============================================================================
    # Basin hopping method for search
    #==============================================================================
    # use method Nelder-Mead 
    #minimizer_kwargs = dict(method="Nelder-Mead", bounds=bnds, options={'disp': True, 'maxiter' : 10})
    #res = sc.optimize.basinhopping(pesObj.getSurvT, x0, niter=100, minimizer_kwargs=minimizer_kwargs, stepsize = 0.1)
    #==============================================================================
    
    
    #==============================================================================
    # Genetic algorithm 
    #==============================================================================
    res = sc.optimize.differential_evolution(pesObj.getSurvT, bnds, maxiter=1)
    #==============================================================================
    np.save(os.getcwd()+'\\parameters'+str(pesObj.cnt),res.x)
    
    pesObj.cnt = pesObj.cnt + 1
    
    
    
    parameter[0:4] = res['x']#res.x
    pesObj.trainNew(parameter)
    
    
    
    #%%
    
    #parameter = [ 0.29607883,  0.40236974,  0.75202162,  0.85884586] # basin hopping
    #parameter = [ 2.40541772,  1.88780256,  1.45359975,  2.20554292] # genetic algorithm
    #parameter = [ 2.47623101,  2.30624922,  1.29450353,  2.12921935] # genetic algorithm 2
    #getSurvT(parameter)

    




