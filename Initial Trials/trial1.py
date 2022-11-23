# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:49:18 2017

@author: rajag038
"""

import gym
env = gym.make('CartPole-v1')
#%%
observation = env.reset()

#%%
env.render()
action = env.action_space.sample()
print(action)
env.step(0)

#%%
for i_episode in range(1):
    observation = env.reset()
    for t in range(1):
        env.render()

        action = env.action_space.sample()
        [observation, reward, done, info] = env.step(action)
        print(observation)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break