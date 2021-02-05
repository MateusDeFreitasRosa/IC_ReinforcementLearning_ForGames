# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 11:10:15 2021

@author: mateu
"""
from collections import deque

lista = deque(maxlen=5)

for i in range(8):
    lista.append(i)
    
print(lista)

K=3
out = []
for i in range(len(lista)-1, len(lista)-K-1, -1):
    out.append(lista[i])
    
print(out)