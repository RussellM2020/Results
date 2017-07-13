import sys
path = '/home/russellm/Research/Results/Split'
sys.path.append(path)
from fileHandling import read
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
_cur_ = '/SplitImage/TRPO'

batches = 200
Rewards1  = read(path + _cur_ + "/trpo40_largeBatch/AvgDisReturn.dat")[:batches]
Rewards2 =  read(path + _cur_ + "/AvgDisReturn.dat")[:batches]
Rewards3 = read(path + _cur_ + "/trpoSplit40/batch10000/AvgDisReturn.dat")[:batches]
#Rewards2 = read(path + _cur_)
#Rewards2 = read(path +_cur_+ "/SL&actOff30/seed40/AvgDisReturn.dat")[:batches]
#Rewards2 = read(path +_cur_+ "/trpoSplit30/rate_e-4/DisRewMean.dat")
#Rewards2 = read(path +_cur_+ "/TRPO/exp_3/try2/AvgDisReturn.dat")
#Sloss = read(path + _cur_ + "/TRPO/exp_3/SLloss.dat")
#rew1 = 0.5*np.add(Rewards1,Rewards2)

#rew2 = read(path + _cur_ + "/SLoff30/DisRewMean.dat")

#rew2 = Rewards2[:batches]
#sl = Sloss
#print(Sloss)

#rew1 = Rewards1[:batches]
#rewSplit = Rewards2[:batches]
# print(batches)

# Rewards = Rewards[:batches]
# print(len(Rewards))

#batches = len(Rewards1)
#print(batches)

Iterations = range(batches)


plt.xlabel("Iterations")
plt.ylabel("Sum of Rewards")
    

plt.plot(Iterations, Rewards1, '-r', label  = 'trpo')
plt.plot(Iterations, Rewards2, '-b', label = 'Split: Rate e-5')
plt.plot(Iterations, Rewards3, '-g', label = 'Split: Rate e-4')

#plt.plot(Iterations, rew1, '-k', label  = 'PG on : 2 fc')
#plt.plot(Iterations, sl, '-g', label  = 'TRPO Split')
    

plt.axis([0, batches,-2*10**4, -100])
plt.title("trpo")
plt.legend()
    
#plt.show()
plt.savefig("Rew_batch10000.png")

