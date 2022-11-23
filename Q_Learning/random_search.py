# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:23:23 2017

@author: rajag038
"""

# ==============================================================================
# Imports
# ==============================================================================
import random
import numpy as np
import matplotlib.pyplot as plt
import time as time
from numpy import linalg as LA

import gym

from collections import deque
import json
# import h5py
from keras.models import Sequential
from keras.models import model_from_json
# from keras.models import h5py
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

# import tensorflow as tf


# ==============================================================================


# ==============================================================================
# Program Constants
# ==============================================================================

# OBSERVATION = 10000 # Timesteps to observe before training
#
# GAMMA = 0.99 # Decay rate of past observations
#
# #-- Exploration - Explotiation balance --#
# EXPLORE = 500000 # Frames over which to anneal epsilon
# FINAL_EPSILON = 0.05 # Final value of epsilon
# INITIAL_EPSILON = 0.3 # Starting value of epsilon
#
# #-- Training parameters --#
TRAIN_INTERVAL = 1
# REPLAY_MEMORY = 200000 # Number of previous transitions to remember
BATCH = 1  # Size of minibatch
# FRAME_PER_ACTION = 1
# LEARNING_RATE = 1e-4


# -- Reward selection --#
REWARD_LOSS = -1
REWARD_NOLOSS = 0.1

ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations

OBSERVATION = 5000.  # timesteps to observe before training

EXPLORE = 1000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4


# ==============================================================================


# ==============================================================================
# Building Q-Function model structure
# ==============================================================================
# def build_model():
#    print("Now we build the model")
#    model = Sequential()
#    model.add(Dense(4, input_dim = 4, activation = 'relu'))
#    model.add(Dense(8, activation = 'relu'))
#    model.add(Dense(8, activation = 'relu'))
#    model.add(Dense(2, activation = 'relu'))
#
#    adam = Adam(lr=LEARNING_RATE)
#    model.compile(loss='mse',optimizer=adam)
#    #model.compile(loss = 'mse', optimizer = 'sgd')
#   print("We finish building the model")
#    return model


def build_model():
    model = Sequential()
    model.add(Dense(4, activation='tanh', input_dim=4))
    ## model.add(Dense(4, activation = ‘tanh’))
    model.add(Dense(2, activation='linear'))

    model.compile(loss='mse', optimizer='sgd')

    return model


# ==============================================================================



# ==============================================================================
# Training the network
# ==============================================================================
def train_network(model, env, init_s, modelName):
    # -- Program Constants --#
    ACTIONS = env.action_space.n  # Number of valid actions
    REND = 0
    RECORD_DIV = 1000
    # -----------------------------------------------------#

    # -- Variable initializations --#
    done = False
    t = 0
    lclT = 0
    r_t = 0
    a_t = 0

    loss = 0
    Q_sa = 0
    rcdCnt = 0
    Q_Arr = np.zeros(np.int(EXPLORE / RECORD_DIV))
    Loss_Arr = np.zeros(np.int(EXPLORE / RECORD_DIV))

    s_t = init_s
    s_t = np.array(s_t.reshape(1, s_t.shape[0]))
    survival_time = 0

    # -- Storage for replay memory --#
    D = deque()

    # -- Exploration of the game state space begins --#
    epsilon = INITIAL_EPSILON

    start_time = time.time()

    while t < EXPLORE:

        if done:
            s_t = env.reset()
            s_t = np.array(s_t.reshape(1, s_t.shape[0]))

        # -- Choosing an epsilong greedy action --#
        if np.random.random() <= epsilon:
            a_t = env.action_space.sample()
        else:
            q = model.predict(s_t)
            max_Q = np.argmax(q)
            a_t = max_Q

        # -- Exploration annealing --#
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # observation, reward, done, info = env.step(action)
        [s_t1, r_t, done, info] = env.step(a_t)

        if (done == 1):
            survival_time = lclT
            if t % RECORD_DIV == 0:
                print('survival time')
                print(lclT)

            if (lclT >= 499):
                r_t = REWARD_NOLOSS
                lclT = 0
            else:
                r_t = REWARD_LOSS
                lclT = 0
        else:
            r_t = REWARD_NOLOSS
            lclT = lclT + 1

        s_t1 = np.array(s_t1.reshape(1, s_t1.shape[0]))

        D.append([s_t, a_t, r_t, s_t1, done])

        # -- Update graphics based on action taken --#
        if (REND == 1):
            env.render()

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # -- Training after the initial observation is complete --#
        if ((t > OBSERVATION) & (t % TRAIN_INTERVAL == 0)):

            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t1.shape[1]))

            targets = np.zeros((inputs.shape[0], ACTIONS))

            for batchCnt in range(0, len(minibatch)):
                state_t = minibatch[batchCnt][0]
                action_t = minibatch[batchCnt][1]
                reward_t = minibatch[batchCnt][2]
                state_t1 = minibatch[batchCnt][3]
                terminal = minibatch[batchCnt][4]

                inputs[batchCnt:batchCnt + 1] = state_t
                targets[batchCnt] = model.predict(state_t)
                Q_sa = model.predict(state_t1)

                # -- Bellman-Deep Q update equations --#
                if terminal:
                    targets[batchCnt, action_t] = reward_t
                else:
                    targets[batchCnt, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss = model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        if (survival_time > 499):
            print('Saving Model')
            model.save_weights(modelName + ".h5", overwrite=True)
            with open(modelName + ".json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            t = EXPLORE
        elif t == EXPLORE:
            print('Saving Model')
            model.save_weights(modelName + ".h5", overwrite=True)
            with open(modelName + ".json", "w") as outfile:
                json.dump(model.to_json(), outfile)



                # -- Saving progress every 1000 iterations --#
                #        if t % RECORD_DIV == 0:
                #            print('Saving Model')
                #            model.save_weights(modelName+".h5", overwrite = True)
                #            with open(modelName+".json", "w") as outfile:
                #                json.dump(model.to_json(), outfile)

                # Local heuristic to stop if sufficiently high 'Q' is reached
                #            if(survival_time>499):
                #                t = EXPLORE

        # -- Print updates of progress every 1000 iterations --#
        if t % RECORD_DIV == 0:
            Q_Arr[rcdCnt] = np.max(Q_sa)
            Loss_Arr[rcdCnt] = loss
            rcdCnt = rcdCnt + 1

            print("TIMESTEP", t, "/ EPSILON", np.round(epsilon,3), "/ ACTION", a_t, "/ REWARD", r_t,  "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    end_time = time.time()
    print('Execution time')
    print(end_time - start_time)
    print('Time per iteration')
    print((end_time - start_time) / EXPLORE)

    return (Q_Arr, Loss_Arr)


# ==============================================================================


# ==============================================================================
# Loading a trained model
# ==============================================================================
def load_model(model, file_name):
    model.load_weights(file_name)

    # model.compile(loss = 'mse', optimizer = 'sgd')
    return model


# ==============================================================================



# ==============================================================================
# Select between model training and evaluation
# ==============================================================================
def deepQ(select, game, modelName, parameter):
    Q_Arr = 0
    Loss_Arr = 0
    if (select == 'Train'):
        model = build_model()
        env = gym.make(game)
        # env.env.gravity=9.8*(0.5)
        #env.env=parameter
        env.env.gravity = parameter[0]
        env.env.masscart = parameter[1]
        env.env.masspole = parameter[2]
        env.env.length = parameter[3]  # actually half the pole's length
        env.env.force_mag = parameter[4]
        env.env.tau = parameter[5]
        init_s = env.reset()
        [Q_Arr, Loss_Arr] = train_network(model, env, init_s, modelName)
        result=1

        plt.plot(Q_Arr)
        plt.figure()
        plt.plot(Loss_Arr)

    elif (select == 'Test'):

        model = build_model()
        env = gym.make(game)
        #env.env.force_mag = 10.0 * 1.5
        #env.env = parameter
        # env.env.gravity=9.8*(0.5)
        # env.env.gravity=24.78
        env.env.gravity = parameter[0]
        env.env.masscart = parameter[1]
        env.env.masspole = parameter[2]
        env.env.length = parameter[3]  # actually half the pole's length
        env.env.force_mag = parameter[4]
        env.env.tau = parameter[5]


        file_name = modelName + ".h5"
        load_model(model, file_name)

        # -- Evaluation --#
        avgT = np.zeros(10)
        avgTR = np.zeros(10)
        for i_episode in range(10):
            observation = env.reset()
            t = 0
            done = 0
            while (done == 0):
                #env.render()
                ## play the game with model
                s_t = observation.reshape(1, observation.shape[0])
                q = model.predict(s_t)
                action = np.argmax(q)
                observation, reward, done, info = env.step(action)
                t = t + 1
                if done:
                    print("Survival time =  {} timesteps".format(t))
                    avgT[i_episode] = t

        # for i_episode in range(10):
        #     observation = env.reset()
        #     t = 0
        #     done = 0
        #     while (done == 0):
        #         env.render()
        #
        #         ## play the game randomly
        #         action = env.action_space.sample()
        #         observation, reward, done, info = env.step(action)
        #         t = t + 1
        #         if done:
        #             print("Survival time =  {} timesteps".format(t))
        #             avgTR[i_episode] = t
        print("\n")
        print("Average Peformances")
        print("Average RL survival time = {} timesteps".format(np.mean(avgT)))
        print("Average Random survival time = {} timesteps".format(np.mean(avgTR)))
        #print("Percentage improvement = {}".format((np.mean(avgT) * 100) / (np.mean(avgTR))))
        #print(model.get_weights())
        result=np.mean(avgT)
    #return (Q_Arr, Loss_Arr)
    return(result)


# ==============================================================================
# class parameter:
#     def __init__(self):
#
#         self.gravity = 9.8
#         self.masscart = 1.0
#         self.masspole = 0.1
#         self.total_mass = (self.masspole + self.masscart)
#         self.length = 0.5  # actually half the pole's length
#         self.polemass_length = (self.masspole * self.length)
#         self.force_mag = 10.0
#         self.tau = 0.02


# ==============================================================================
# Main function area
# ==============================================================================

# parameter.gravity = 9.8
# parameter.masscart = 1.0
# parameter.masspole = 0.1
# parameter.total_mass = (parameter.masspole + parameter.masscart)
# parameter.length = 0.5  # actually half the pole's length
# parameter.polemass_length = (parameter.masspole * parameter.length)
# parameter.force_mag = 10.0
# parameter.tau = 0.02
parameter=np.zeros(6)
parameter[0]=9.8
parameter[1]=1.0
parameter[2]=0.1
parameter[3]=0.5
parameter[4]=10.0
parameter[5]=0.02

model_name='model1'
baseline=deepQ('Test', 'CartPole-v1', 'model1',parameter)
#for num in range(10):
num=1
Max_iter=10
while(num<Max_iter):
    sign=0
    for i in [0,3,5]:
        ratio=0
        temp=parameter[i]
        parameter[i]=1.2*temp
        result_old1=deepQ('Test', 'CartPole-v1', model_name,parameter)
        parameter[i]=0.2*temp
        result_old2=deepQ('Test', 'CartPole-v1', model_name,parameter)
        parameter[i] = temp

        if ((result_old1>=baseline) and (result_old2>=baseline)):
            sign=0
        elif result_old1-result_old2<0:
            polarity=0.1*temp
            t = 0
            while ((ratio < 0.1) and (t < 10) and (np.absolute(parameter[i] - temp) < 0.5 * temp) and (
                parameter[0] > 0)):  # range of the parameter
                # [Q_Arr, Loss_Arr] = deepQ('Test', 'CartPole-v1', 'model_f15',parameter)
                t = t + 1
                parameter[i] = parameter[i] + polarity
                result_new = deepQ('Test', 'CartPole-v1', model_name, parameter)
                ratio = np.absolute(result_new - baseline) / baseline
            if parameter[i] != temp:
                sign = 1
        else:
            polarity=-0.1*temp
            t = 0
            while ((ratio < 0.1) and (t < 10) and (np.absolute(parameter[i] - temp) < 0.5 * temp) and (
                parameter[0] > 0)):  # range of the parameter
                # [Q_Arr, Loss_Arr] = deepQ('Test', 'CartPole-v1', 'model_f15',parameter)
                t = t + 1
                parameter[i] = parameter[i] + polarity
                result_new = deepQ('Test', 'CartPole-v1', model_name, parameter)
                ratio = np.absolute(result_new - baseline) / baseline
            if parameter[i] != temp:
                sign = 1
    if sign==1:
        print('train a new model')
        num = num + 1
        model_name='model'+str(num)
        baseline=deepQ('Train','CartPole-v1',model_name,parameter)
        baseline = deepQ('Test', 'CartPole-v1', model_name, parameter)
    else:
        Max_iter=num

print(parameter)


# %%




# model = build_model()
# model = load_model(model, 'model_g15.h5')
# weights = model.get_weights()
# variable = np.asarray(weights[0])
# for i in [1, 2, 3]:
#     a = np.asarray(weights[i])
#     variable = np.append(variable, a)
#
# model = build_model()
# model = load_model(model, 'model_g05.h5')
# weights = model.get_weights()
# variable1 = np.asarray(weights[0])
# for i in [1, 2, 3]:
#     a = np.asarray(weights[i])
#     variable1 = np.append(variable1, a)
#
# diff = LA.norm(variable - variable1) / 30






# #%%
# print weights
# ==============================================================================
