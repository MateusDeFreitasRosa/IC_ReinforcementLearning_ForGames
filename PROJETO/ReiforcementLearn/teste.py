import gym
from collections import deque
import random
import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np
import sys

data = deque(maxlen=2000000)
env = gym.make('Breakout-v0')
env.reset()

for i in range(2000000):
  action = random.randrange(4)
  next_state, reward, done, info = env.step(action)
  data.append((next_state, reward, done, info))



state_save = []
reward_save = []
done_save = []
info_save = []

for i in data:
    state_save.append(i[0])
    reward_save.append(i[1])
    done_save.append(i[2])
    info_save.append(i[3])

ds = pandas.DataFrame(data={'state':state_save, 'reward': reward_save, 'done': done_save, 'info': info_save}).to_pickle('./file.csv')


read = pandas.read_pickle('./file.csv')
firstState = read['state'][100]


plt.imshow(firstState)