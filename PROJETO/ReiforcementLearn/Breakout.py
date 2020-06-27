# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:10:11 2020

@author: mateu
"""

import gym
import cv2
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import os
import random
import matplotlib.pyplot as plt

score = []

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "breakout.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=5)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()
        self.k_frames           = 4

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
            
        #Criar camada de convolução.
        model.add(Conv2D(32, (8,8), strides=4,  input_shape = (84,84,1) , activation = 'relu', padding='same'))
        model.add(MaxPooling2D(pool_size = (2,2), dim_ordering='th'))
        model.add(Conv2D(64, (4,4), strides=2, activation = 'relu', padding='same'))
        model.add(MaxPooling2D(pool_size = (2,2), dim_ordering='th'))
        model.add(Conv2D(64, (3,3), strides=1, activation = 'relu', padding='same'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #model.add(Conv2D(256, (3,3), activation = 'relu', padding='same'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Flatten())
        
        #Criação da rede Neural.
        model.add(Dense(76, activation='relu'))
        #model.add(Dropout(.5))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model
    
    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))

    
    def packK_frames(self):
        if len(self.memory) < self.k_frames+5:
            return
        index = random.randint(0, len(self.memory)-self.k_frames-5)
        state = np.empty((self.k_frames, ))


    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay



class Breakout():
    
    def __init__(self):
        self.env = gym.make('Breakout-v0')
        self.sample_batch_size = 4
        self.episodes = 2000
        
        self.action_size = self.env.action_space.n
        self.state_size = (84,84)
        self.agent = Agent(self.state_size, self.action_size)

    
    def to_gray_scale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)
    
    def down_sample(self, img):
        return img[::2,::2]
    
    def resize(self, img):
        return cv2.resize(self.to_gray_scale(self.down_sample(img)), (84,84)) / 255
        
    def transform_reward(self, reward):
        return np.sign(reward)

    def run(self):
        number_img = 0
        try:
            for i_episodes in range(self.episodes):
                state = self.env.reset()
                
                state = self.resize(state)
                state = np.expand_dims(state, axis=2)
                state = np.expand_dims(state, axis=0)
                
                done = False
                index=0
                total_reward=0
                while not done:
                    if i_episodes > 1000:
                        self.env.render()
                    
                    
                    number_img+=1
                    action = self.agent.act(state)
                    
                    next_state, reward, done, info = self.env.step(action)
                
                    reward = np.sign(reward)
                    total_reward+=reward
                    next_state = self.resize(next_state)
                    if i_episodes > 300:
                        plt.savefig( 'images/'+str(number_img)+'.png', format='png')
                        plt.imshow(next_state, filternorm=3)
                        plt.show()
                        
                    next_state = np.expand_dims(next_state, axis=2)
                    next_state = np.expand_dims(next_state, axis=0)
                    
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index+=1
                
                #self.time.add_list_time()
                #out_consult_time = self.time.consult_time()
                #out_consult_time_current = self.time.consult_time_current()
                
                print("Episode {}# Rewards: {}".format(i_episodes, reward))
                score.append(index)
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()
            #self.time.compare_begin_with_end_time()
        
            
if __name__ == '__main__':
    breakout = Breakout()
    breakout.run()      
    
