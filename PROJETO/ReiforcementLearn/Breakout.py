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

####VERIFY IF GPU IS RUNNING.
#print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
import keras

import tensorflow.compat.v1 as tf1

config = tf1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)


######################################################################################################################################

import numpy as np
import tensorflow as tf

'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

score = []

TRAIN = True
OBSERVER = 5000
UPDATE_TARGET_MODEL = 1000
qntUpdate=0

class Agent():
    def __init__(self, state_size, action_size,):
        self.weight_backup      = "breakout_dataModel.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=50000) if TRAIN else deque(maxlen=5)
        self.learning_rate      = 0.00005
        self.gamma              = 0.99
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 1e-5
        self.k_frames           = 4
        self.frame_height       = self.state_size[0]
        self.frame_width        = self.state_size[1]
        self.brain              = self._build_model()
        self.brain_target       = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
            
        #Criar camada de convolução.
        #-> (32,84,84,4)
        model.add(Conv2D(32, (8,8), strides=4,  input_shape = (self.state_size[0], self.state_size[1], self.k_frames) , activation = 'relu', padding='valid'))
        #<- (84,84)
        #->(84,84)
        #model.add(MaxPooling2D(pool_size = (2,2), dim_ordering='th'))
        #<-(42,42)
        #->(42,42)
        model.add(Conv2D(64, (4,4), strides=2, activation = 'relu', padding='valid'))
        #<-(42,42)
        #->(42,42)
        #model.add(MaxPooling2D(pool_size = (2,2), dim_ordering='th'))
        #<-(21,21)
        #->(21,21)
        model.add(Conv2D(64, (3,3), strides=1, activation = 'relu', padding='valid'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        #<-(11,11)
        #model.add(Conv2D(256, (3,3), activation = 'relu', padding='same'))
        #model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Flatten())
        
        #Criação da rede Neural.
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        #model.add(Dropout(.5))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model
    
    def save_model(self):
            self.brain.save(self.weight_backup)

    def get_last_k_frames(self, state):
        frames = np.empty((self.k_frames, self.frame_height, self.frame_width))

        for i in range(0, self.k_frames):
            s, _, _, _, _ = self.memory[len(self.memory)-(self.k_frames-i)]
            frames[i] = s
            
        frames[3] = state  
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
        predicted = self.brain_target.predict(next_state) #Previsão proximo estado.
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
        history = self.brain.fit(state, target_f, batch_size=sample_batch_size, epochs=1, verbose=0)
        
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate -= self.exploration_decay
    
        return history, self.exploration_rate

    def update_target_model(self, frame_number):
        if (frame_number % UPDATE_TARGET_MODEL == 0 and TRAIN and frame_number > OBSERVER):
            self.brain_target.set_weights(self.brain.get_weights())
            global qntUpdate
            qntUpdate+=1
            print('-------------------------UPDATED TARGET MODEL({}) ------------------------------'.format(qntUpdate))

data_average_reward = []
total_reward_game = []
class Breakout():
    
    def __init__(self):
        self.env = gym.make('Pong-v0')
        self.sample_batch_size = 32
        self.episodes = 220000
        
        self.action_size = self.env.action_space.n
        self.state_size = (80,80)
        self.agent = Agent(self.state_size, self.action_size)
        self.number_print = 0
        self.best_score = 0
        self.replay_bestPlay = deque(maxlen=300)
        self.crop_on_top = 35 #57
        self.crop_on_bottom = 15
        self.crop_on_border = 7
        self.frame_number=0
        self.freq_update=4
        self.qnt_train=0
    
    
    def to_gray_scale(self, img):
        return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    
    def down_sample(self, img):
        return img[::2,::2]
    
    def preprocess_img(self, img):
        return np.asarray(self.to_gray_scale(self.down_sample(self.crop_img(img))) / 255, dtype=np.float32)
    
    def crop_img(self, img):
        #return img[self.crop_on_top: -self.crop_on_bottom, self.crop_on_border:-self.crop_on_border]
        return img[35:195]
  
    def transform_reward(self, reward):
        return np.sign(reward)
      
    
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
               
                state = self.preprocess_img(state)
            
                done = False
                index=0
                total_reward=0
                history_list = []
                exploration = 1.0
                while not done:

                    if i_episodes > 300:
                        self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    #if (info['ale.lives'] < 5):
                    #    done = True
                    
                    reward = np.sign(reward)
                    total_reward+=reward
                    #self.replay_bestPlay.append(next_state)
                    #if total_reward >= 5:
                    #    self.smile_for_the_photo()
                        
                    next_state = self.preprocess_img(next_state)
                    
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index+=1
                    
                    self.frame_number+=1
                    if (TRAIN and self.frame_number > OBSERVER):
                        history, exploration = self.agent.replay(self.sample_batch_size)
                        history_list.append(history.history['loss'])
                        self.qnt_train+=1
                        
                    self.agent.update_target_model(self.frame_number)
                #self.time.add_list_time()
                #out_consult_time = self.time.consult_time()
                #out_consult_time_current = self.time.consult_time_current()
                total_reward_game.append(total_reward)
                data_average_reward.append(np.mean(total_reward_game[-50:]))
               
                if total_reward > self.best_score:
                    self.best_score = total_reward
                
                score.append(index)
                #if TRAIN:
                #    history = self.agent.replay(self.sample_batch_size)
                print("Episode {}# Rewards: {}# Average: {:.3}# Loss: {:.6} # Train: {} # Exploration: {:.3}".format(i_episodes, total_reward, data_average_reward[len(data_average_reward)-1], np.mean(history_list), self.qnt_train, exploration))
        finally:
            plt.plot(data_average_reward)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
        
            if TRAIN:
                self.agent.save_model()
            #self.time.compare_begin_with_end_time()
            self.env.close()
            
if __name__ == '__main__':
    breakout = Breakout()
    breakout.run()      
    
    
        
    
    
    
    
    
    
