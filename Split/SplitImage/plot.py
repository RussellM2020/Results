import sys
path = '/home/russellm/Research/Results/Split'
sys.path.append(path)
from fileHandling import read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_cur_ = '/SplitImage'



Rewards = read(path +_cur_+ "/TRPO/exp_12/AvgDisReturn.dat")
#Rewards_sloff = read(path +_cur_+ "/TRPO/exp_11/AvgDisReturn.dat")



batches = len(Rewards)
# print(batches)

# Rewards = Rewards[:batches]
# print(len(Rewards))
Iterations = range(batches)


plt.xlabel("Iterations")
plt.ylabel("Sum of Rewards")
    
plt.plot(Iterations, Rewards, '-r', label  = 'SL on')
#plt.plot(Iterations, Rewards_sloff, '-b', label  = 'SL off')
    

plt.axis([0, batches,-9*10**3, -2*10**3])
plt.title("trpo")
plt.legend()
    
#plt.show()
plt.savefig("rew.png")

