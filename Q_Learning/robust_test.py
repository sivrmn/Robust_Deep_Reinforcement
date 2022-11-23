#==============================================================================
# Imports
#==============================================================================
import random
import numpy as np
import matplotlib.pyplot as plt
import time as time
import csv


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



#==============================================================================
#%%


#==============================================================================
# Program Constants
#==============================================================================

OBSERVATION = 10000 # Timesteps to observe before training

GAMMA = 0.99 # Decay rate of past observations

#-- Exploration - Explotiation balance --#
EXPLORE = 500000 #50000 # Frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # Final value of epsilon
INITIAL_EPSILON = 0.2 # Starting value of epsilon

#-- Training parameters --#
TRAIN_INTERVAL = 1
REPLAY_MEMORY = 100000 # Number of previous transitions to remember
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
    print("Now we build the model")
    model = Sequential()
    model.add(Dense(4, input_dim = 4, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(2, activation = 'relu'))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)    
    #model.compile(loss = 'mse', optimizer = 'sgd')
    print("We finish building the model")
    return model    
#==============================================================================

#==============================================================================
# Loading a trained model
#==============================================================================
def load_model(model, file_name):
    model.load_weights(file_name)
    model.compile(loss = 'mse', optimizer = 'sgd')
    return model    
#==============================================================================



def robust_test(model, env, w, test_length):
    ## w is a length 6 vector to change the parameters by percents
    ## gravity, masscart, masspole, length, force_mag, tau, (theta_threshold_radians)
    p_dist_free = []
    p_dist = []
    for i_episode in range(test_length):
        observation = env.reset()
        for t in range(1000):
            ## env.render()

            s_t = observation.reshape(1, observation.shape[0])
            q = model.predict(s_t)
            action = np.argmax(q)
            observation, reward, done, info = env.step(action)

            if done:
                p_dist_free.append(t)
                break

    env.env.gravity = env.env.gravity * (1 + w[0])
    env.env.masscart = env.env.masscart * (1 + w[1])
    env.env.masspole = env.env.masspole * (1 + w[2])
    env.env.length = env.env.length * (1 + w[3])
    env.env.force_mag = env.env.force_mag * (1 + w[4])
    env.env.tau = env.env.tau * (1 + w[5])
##    env.env.theta_threshold_radians = env.env.theta_threshold_radians * (1 + w[6])

    for i_episode in range(test_length):
        observation = env.reset()
        for t in range(1000):
            ## env.render()

            s_t = observation.reshape(1, observation.shape[0])
            q = model.predict(s_t)
            action = np.argmax(q)
            observation, reward, done, info = env.step(action)

            if done:
                p_dist.append(t)
                break

    ave_p_df = np.mean(p_dist_free)
    ave_p_d = np.mean(p_dist)
    print("Mean survival steps without disturbance: ", ave_p_df)
    print("Mean survival steps with disturbance: ", ave_p_d)
    print("diff-disturbance ratio: ", ((ave_p_df - ave_p_d) / ave_p_df) / np.mean(w))

    line1, = plt.plot(p_dist_free, label = "DisFree")
    line2, = plt.plot(p_dist, label = "Dis")
    plt.axis([0, test_length, min(p_dist) - 50, 1100])
    plt.legend()
    
    plt.show()
            
def robust_trend(model, env, start, end, test_length):
    ## disturbance will be given from start to end
    start = np.round(start, 1)
    end = np.round(end, 1)
##    trials = int((end - start) / 0.1)
    trials = 200
    p_dist_free = 999.0
    data = []
    w = start
    ddr = []
    dist = []
    dataArr = {}
    for i in range(trials):
        p_dist = 0.
        env_0 = gym.make('CartPole-v1')
        env_0.env.gravity = env_0.env.gravity * (1 + random.uniform(-w, w))
        env_0.env.masscart = env_0.env.masscart * (1 + random.uniform(-w, w))
        env_0.env.masspole = env_0.env.masspole * (1 + random.uniform(-w, w))
        env_0.env.length = env_0.env.length * (1 + random.uniform(-w, w))
        env_0.env.force_mag = env_0.env.force_mag * (1 + random.uniform(-w, w))
        env_0.env.tau = env_0.env.tau * (1 + random.uniform(-w, w))

        for i_episode in range(test_length):
            observation = env_0.reset()
            for t in range(1000):
                s_t = observation.reshape(1, observation.shape[0])
                q = model.predict(s_t)
                action = np.argmax(q)
                observation, reward, done, info = env_0.step(action)

                if done:
                    p_dist = p_dist + t
                    break

        p_dist = p_dist / test_length
        diff_dist_ratio = ((p_dist_free - p_dist) / p_dist_free) / abs(w)
        data.append([diff_dist_ratio, env_0.env.gravity, env_0.env.masscart, env_0.env.masspole, 
                     env_0.env.length, env_0.env.force_mag, env_0.env.tau])
        ddr.append(diff_dist_ratio)
        dist.append(w)

##        w = w + 0.1
        dataArr[i] = data

    head = ["ddr", "gravity", "masscart", "masspole", "length", "force_mag", "tau"]
    with open('diff_dist_rat.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(head)
        for i in range(trials):
          spamwriter.writerow(data[i])
    return(dataArr)
          
          
#%%
modelName = 'modelrob4'
game = 'CartPole-v1'

model = build_model()
env = gym.make(game)

file_name = modelName+".h5"
model = load_model(model, file_name)

start = 0.2
end = 0.5
test_length =10
w = np.array([1,1,1,1,1,1])*0.5
#robust_trend(model, env, start, end, test_length)
robust_test(model, env, w, test_length)
