# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:42:35 2020

@author: mateu
"""

from python_server_FCEUX import Server
import matplotlib.pyplot as plt
import json
import numpy as np

server = Server()

image = server.sendCommandAndReceiveOperation('getScreenShot')
print('LenImage: {}'.format(len(image)))
image = image.decode('utf-8').split('json')[1]
image