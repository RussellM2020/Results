import sys
path = '/home/russellm/Research/Results/Split'
sys.path.append(path)
from fileHandling import read
from fileHandling import store
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_cur_ = '/SplitImage/TRPO/trpoSplit40'

rewStore = []
batches = 300

Rewards1 = read(path +_cur_+ "/seed3/AvgDisReturn.dat")[:batches]
Rewards2 = read(path +_cur_+ "/seed15/AvgDisReturn.dat")[:batches]
Rewards3 = read(path +_cur_+ "/seed31/AvgDisReturn.dat")[:batches]
#Rewards2 = read(path +_cur_+ "/TRPO/exp_3/try2/AvgDisReturn.dat")
#Sloss = read(path + _cur_ + "/TRPO/exp_3/SLloss.dat")

rewStore.append(Rewards1)
rewStore.append(Rewards2)
rewStore.append(Rewards3)

Mean = np.mean(rewStore, axis = 0)
Std = np.std(rewStore, axis =0)

#sl = Sloss
#print(Sloss)

#rew1 = Rewards1[:batches]
#rewSplit = Rewards2[:batches]
# print(batches)

# Rewards = Rewards[:batches]
# print(len(Rewards))
store(path+_cur_+"/DisRewMean.dat", Mean)
store(path+_cur_+"/DisRewSTD_seeded.dat", Std)