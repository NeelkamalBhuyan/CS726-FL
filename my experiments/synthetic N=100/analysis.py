#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:05:47 2022

@author: nkb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colors = ['blue', 'red', 'limegreen', 'black']
line = ['dashed', 'solid','dashed', 'dotted']

for i in range(4):
    df = pd.read_csv("policy"+str(i)+".csv")
    plt.plot(range(0,101,10),df['model train acc'], color = colors[i], linestyle = line[i])

plt.legend(['FedAvg', 'policy 1', 'policy 2', 'policy 3'])
plt.grid()
plt.xlabel("number of rounds")
plt.ylabel("training acc")

plt.figure()
for i in range(4):
    df = pd.read_csv("policy"+str(i)+".csv")
    plt.plot(range(0,101,10),df['model test acc'], color = colors[i], linestyle = line[i])


plt.legend(['FedAvg', 'policy 1', 'policy 2', 'policy 3'])
plt.grid()
plt.xlabel("number of rounds")
plt.ylabel("training acc")
