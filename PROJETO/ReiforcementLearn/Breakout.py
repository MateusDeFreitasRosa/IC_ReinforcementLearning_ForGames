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

######################################################################################################################################
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

######################################################################################################################################


score = []

TRAIN = True

class Agent():
    def __init__(self, state_size, action_size, frame_height, frame_width):
        self.weight_backup      = "breakout_dataModel.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=100000) if TRAIN else deque(maxlen=5)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.k_frames           = 4
        self.frame_height       = frame_height
        self.frame_width        = frame_width
        self.brain              = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
            
        #Criar camada de convolução.
        #-> (32,84,84,4)
        model.add(Conv2D(32, (8,8), strides=4,  input_shape = (self.frame_height, self.frame_width, self.k_frames) , activation = 'relu', padding='same'))
        #<- (84,84)
        #->(84,84)
        #model.add(MaxPooling2D(pool_size = (2,2), dim_ordering='th'))
        #<-(42,42)
        #->(42,42)
        model.add(Conv2D(64, (4,4), strides=2, activation = 'relu', padding='same'))
        #<-(42,42)
        #->(42,42)
        #model.add(MaxPooling2D(pool_size = (2,2), dim_ordering='th'))
        #<-(21,21)
        #->(21,21)
        model.add(Conv2D(128, (3,3), strides=1, activation = 'relu', padding='same'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        #<-(11,11)
        #model.add(Conv2D(256, (3,3), activation = 'relu', padding='same'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Flatten())
        
        #Criação da rede Neural.
        model.add(Dense(512, activation='relu'))
        #model.add(Dropout(.5))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model
    
    def save_model(self):
            self.brain.save(self.weight_backup)

    def get_last_k_frames(self, state):
        frames = np.empty((self.k_frames, self.frame_height, self.frame_width))

        for i in range(1, 5):
            state, _, _, _, _ = self.memory[len(self.memory)-i]
            frames[i-1] = state
          
        return np.transpose(frames, axes=(1,2,0))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        if len(self.memory) < self.k_frames+1:
            return random.randrange(self.action_size)
        k_frames_state = self.get_last_k_frames(state)
        k_frames_state = np.expand_dims(k_frames_state, axis=0)
        act_values = self.brain.predict(k_frames_state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))

    
    def pack_K_frames(self, sample_batch_size):
        
        if len(self.memory) < self.k_frames+2:
            return
        
        state = np.empty((sample_batch_size, self.k_frames, self.frame_height, self.frame_width))
        action = np.empty(sample_batch_size, dtype=np.uint8)
        reward = np.empty(sample_batch_size, dtype=np.float32)
        next_state = np.empty((sample_batch_size, self.k_frames, self.frame_height, self.frame_width))
        done = np.empty(sample_batch_size, dtype=np.bool)
        
        for k in range(sample_batch_size):
            index = random.randint(0, len(self.memory)-self.k_frames-2)
            for i, idx_memory in enumerate(range(index, index+self.k_frames)):
                s, a, r, n_s, d = self.memory[idx_memory]
            
                state[k][i] = s
                next_state[k][i] = n_s
                
            done[k] = d
            action[k] = a
            reward[k] = r
    
        #State = (32,4,84,84)  -> State Transpose = (32,84,84,4) 
        return np.transpose(state, axes=(0,2,3,1)), action, reward, np.transpose(next_state, axes=(0,2,3,1)), done
        

    def replay(self, sample_batch_size):
       
        if len(self.memory) < sample_batch_size:
            return
        
        #sample_batch = random.sample(self.memory, sample_batch_size)
        state, action, reward, next_state, done = self.pack_K_frames(sample_batch_size)
        
        #print('State: {}'.format(state.shape))
        #print('Action: {}'.format(action.shape))
        #print('Reward: {}'.format(reward.shape))
        #print('Next_State: {}'.format(next_state.shape))
        #print('Done: {}'.format(done.shape))
        #input()
        #target = reward
        predicted = self.brain.predict(next_state) #Previsão proximo estado.
        target_f = self.brain.predict(state) #Previsão estado atual.
        #print('Predicted: {}'.format(predicted))
        #print('Predicted[0]: {}'.format(predicted[2]))
        #print('Predicted Max: {}'.format(np.amax(predicted)))
        #print('Reward: {}'.format(reward))
        #print('Target_f: {}'.format(target_f))
        #input()
        
        
        for i in range(sample_batch_size):
            target = reward[i] + (self.gamma * np.amax(predicted[i]) * (1-done[i]))
            target_f[i][action[i]] = target
            #print('Action: {}'.format(action[i]))
            #print('target: {}'.format(target))
            #print('target_f: {}'.format(target_f[i]))
            #input()
            
        
        #print('Target_f Formatado: {}'.format(target_f))
        #input()    
        
        history = self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
        return history


data_reward = []
class Breakout():
    
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.sample_batch_size = 32
        self.episodes = 220000
        
        self.action_size = self.env.action_space.n
        self.state_size = (84,84)
        self.agent = Agent(self.state_size, self.action_size, 84, 84)
        self.number_print = 0
        self.best_score = 0
        self.replay_bestPlay = deque(maxlen=300)
        self.crop_on_top = 35 #57
        self.crop_on_bottom = 15
        self.crop_on_border = 7
        self.frame_number=0
        self.freq_update=4
    
    
    def to_gray_scale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)
    
    def down_sample(self, img):
        return img[::2,::2]
    
    def resize(self, img):
        return cv2.resize(self.to_gray_scale(self.down_sample(img)), (84,84)) / 255
        
    def transform_reward(self, reward):
        return np.sign(reward)
    
    def crop_img(self, img):
        #return img[self.crop_on_top: -self.crop_on_bottom, self.crop_on_border:-self.crop_on_border]
        return img[14:-15]
        
    
    def smile_for_the_photo(self):
        print('Salvar Imagens (pressione qualquer tecla para continuar)')
        input()
        for i in range(len(self.replay_bestPlay)):
            plt.imshow(self.replay_bestPlay[i])
            plt.savefig( 'images/'+str(self.number_print)+'.png', format='png')
            plt.show()
            self.number_print+=1
        print('Imagens salvas com sucesso!')
        input()

    def run(self):
        
        try:
            for i_episodes in range(self.episodes):
                state = self.env.reset()
                state = self.crop_img(state)
                state = self.resize(state)
            
                done = False
                index=0
                total_reward=0
                while not done:

                    self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    #if (info['ale.lives'] < 5):
                    #    done = True
                    next_state = self.crop_img(next_state)
                    
                    reward = np.sign(reward)
                    total_reward+=reward
                    #self.replay_bestPlay.append(next_state)
                    #if total_reward >= 5:
                    #    self.smile_for_the_photo()
                        
                    next_state = self.resize(next_state)
                    
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index+=1
                   
                #self.time.add_list_time()
                #out_consult_time = self.time.consult_time()
                #out_consult_time_current = self.time.consult_time_current()
                data_reward.append(total_reward)
                if total_reward > self.best_score:
                    self.best_score = total_reward
                
                score.append(index)
                if TRAIN:
                    history = self.agent.replay(self.sample_batch_size)
                print("Episode {}# Rewards: {}# BestScore: {}# Loss: {}".format(i_episodes, total_reward, self.best_score, history.history['loss'] if history != None else '-'))
        finally:
            plt.plot(data_reward)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
        
            if TRAIN:
                self.agent.save_model()
            #self.time.compare_begin_with_end_time()
        
            
if __name__ == '__main__':
    breakout = Breakout()
    breakout.run()      
    
    
        
    
    
    
    
    
    
