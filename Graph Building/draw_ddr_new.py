import csv
import numpy as np
import gym
import matplotlib.pyplot as plt

from collections import deque
import json
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

LEARNING_RATE = 1e-4

def ddr(model, env, w, parameter):
  ## only test one parameter at a time
  ## parameter range from 0 to 3
  ## gravity: 0
  ## masscart: 1
  ## masspole: 2
  ## length: 3
  para_ref = {'gravity':0, 'masscart':1, 'masspole':2, 'length':3, 'force_mag':4, 'tau':5}
  choose_para = [0 for i in range(6)]
  choose_para[para_ref[parameter]] = 1
  
  p_dist_free = []
  p_dist = []
  for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
      s_t = observation.reshape(1, observation.shape[0])
      q = model.predict(s_t)
      action = np.argmax(q)
      observation, reward, done, info = env.step(action)

      if done:
        p_dist_free.append(t)
        break
  
  env.env.gravity = env.env.gravity * (1 + w * choose_para[0])
  env.env.masscart = env.env.masscart * (1 + w * choose_para[1])
  env.env.masspole = env.env.masspole * (1 + w * choose_para[2])
  env.env.length = env.env.length * (1 + w * choose_para[3])
  env.env.force_mag = env.env.force_mag * (1 + w * choose_para[4])
  env.env.tau = env.env.tau * (1 + w * choose_para[5])

  for i_episode in range(10):
    observation = env.reset()
    for t in range(1000):
      s_t = observation.reshape(1, observation.shape[0])
      q = model.predict(s_t)
      action = np.argmax(q)
      observation, reward, done, info = env.step(action)

      if done:
        p_dist.append(t)
        break

  ave_p_df = np.mean(p_dist_free)
  ave_p_d = np.mean(p_dist)
  DDR = ((ave_p_df - ave_p_d) / ave_p_df) / w
  return DDR

def draw_ddr(w_list, ddr_list, parameter, prefix):
  ## only test one parameter at a time
  ## parameter range from 0 to 3
  ## gravity: 0
  ## masscart: 1
  ## masspole: 2
  ## length: 3
  plt.figure()
  line1, = plt.plot(w_list, ddr_list, label = parameter)
  plt.axis([min(w_list), max(w_list), min(ddr_list) - 0.1, max(ddr_list) + 0.1])
  plt.legend()
  plt.title("DDR over disturbance on " + parameter)
  plt.xlabel("disturbance")
  plt.ylabel("DDR")

  plt.savefig(prefix + parameter + '.eps', format = 'eps', dpi = 1000)
  plt.savefig(prefix + parameter + '.png', format = 'png', dpi = 1000)
##  plt.show()

def load_shallow_model_list(model_name_list):
  model_list = []
  for i in model_name_list:
    model = build_shallow_model()
    model.load_weights(i)
    model.compile(loss = 'mse', optimizer = 'sgd')

    model_list.append(model)

  return model_list

def load_shallow_model(model_name):
  model = build_shallow_model()
  model.load_weights(model_name)
  model.compile(loss = 'mse', optimizer = 'sgd')

  return model

def load_deep_model(model_name):
  model = build_deep_model()
  model.load_weights(model_name)
  adam = Adam(lr=LEARNING_RATE)
  model.compile(loss='mse',optimizer=adam)

  return model

def build_shallow_model():
  print("Now we build the model")
  model = Sequential()
  model.add(Dense(4, activation = 'tanh', input_dim = 4))
  model.add(Dense(2, activation = 'linear'))
  
  model.compile(loss = 'mse', optimizer = 'sgd')
  print("We finish building the model")
  return model

def build_deep_model():
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

def restore_env():
  env = gym.make('CartPole-v1')
  return env

def store_data(ddr_list, prefix):
  with open(prefix + '.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in ddr_list:
      spamwriter.writerow(i)

  print "Data Saved!!"

def load_data(filename):
  data = []
  with open(filename, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    for row in spamreader:
      data.append(row)

  print "Data Loaded!!"
  
  return data
      
  

para_name = ['gravity', 'masscart', 'masspole', 'length', 'force_mag', 'tau']
model_name_list = ['model_7.h5', 'ModelNonRobustNew1Good.h5', 'modelrob4.h5', 'model_8.h5', 'model_9.h5']
shallow_name_list = ['model_7.h5', 'model_8.h5', 'model_9.h5']
deep_name_list = ['ModelNonRobustNew1Good.h5' ,'ModelNonRobustNew2Good.h5', 'ModelNonRobustNew3Good.h5']
noise_name = 'modelrob4.h5'
model = load_shallow_model(shallow_name_list[0])
w_list = np.arange(-0.9, 1.1, 0.1)

data = []
for j in range(6):
  ddr_list = []
  for i in w_list:
    env = restore_env()
    DDR = 0 if i > -0.05 and i < 0.05 else ddr(model, env, i, para_name[j])
    ddr_list.append(DDR)

  data.append(ddr_list)
  print data

store_data(data, 'model_7')

  ## draw_ddr(w_list, ddr_list, para_name[j], 'ss9_')
    

