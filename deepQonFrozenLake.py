# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 08:45:50 2019

@author: Surendran Nambiar
"""

import gym
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
import random
from collections import deque
import time


class DQNA:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.gamma=0.95
        self.memory = deque(maxlen=2000)
        self.epsilon = 1
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.995
        self.lr = 0.002
        self.model = self._build_model()
    def _build_model(self):
        model = Sequential()
        model.add(Dense(units=24,input_shape=(self.state_size,),activation='relu'))
        model.add(Dense(units=24,activation='relu'))
        model.add(Dense(units=self.action_size,activation='linear'))
        model.compile(optimizer='adam',loss='mean_squared_error')
        return model
    def remember(self,state,action,reward,next_state,done):
        state1=np.zeros(self.state_size)
        state1[state]=1.00
        state1=np.reshape(state1,(1,16))
        state2=np.zeros(self.state_size)
        state2[next_state]=1.00
        state2=np.reshape(state2,(1,16))
        self.memory.append((state1,action,reward,state2,done))
    def act(self,state):
        prepro=np.zeros(self.state_size)
        prepro[state]=1.00
        prepro=np.reshape(prepro,(1,16))
        if random.uniform(0,1)<self.epsilon:
            return np.random.choice(self.action_size)
        action=self.model.predict(prepro)
        return np.argmax(action)
    def action_replay(self,batch_size):
        minbatch = random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minbatch:
            target = reward
            if not done:
                target = reward+self.gamma*np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state,target_f,epochs=1,verbose=0)
        if self.epsilon >self.epsilon_min:
            self.epsilon *= self.epsilon_decay
env = gym.make('FrozenLake-v0')
agent = DQNA(env.observation_space.n,env.action_space.n)

for i in range(15000):
    state = env.reset()
    for j in range(50):
        action = agent.act(state)
        obs,reward,done,_ = env.step(action)
        agent.remember(state,action,reward,obs,done)
        if done:
            print(i)
            break
    if len(agent.memory)>=32:
        agent.action_replay(32)
        
#play rate
state = env.reset()
env.render()
while True:
    action = agent.act(state)
    obs,reward,done,_ = env.step(action)
    env.render()
    agent.remember(state,action,reward,obs,done)
    time.sleep(1)
    if done:
        break