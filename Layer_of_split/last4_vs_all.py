import matplotlib.pyplot as plt
import numpy as np

from last4_Vpg import cartpoleSplit_4
from dualCopy import cartpoleVpg


batches = 50
Iterations= range(0,batches) 


last4History=[]
allHistory=[]

def rewardPlot(list1,list2):
    plt.clf()
    plt.xlabel("Iterations")
    plt.ylabel("Sum of Rewards")
    
    
    plt.plot(Iterations, list1, '-r', label  = 'Last 4 layers vpg')
    plt.plot(Iterations, list2, '-b', label  = 'all 5 layers vpg')


    plt.axis([0, batches,0, 202])
    plt.title("Reward 1-4 Split vs VPG")
    plt.legend()
    
    plt.savefig("Reward_last4_all " +str(seed)+".png")


for seed in range(0,100,20):
    Rewards4,_,_= cartpoleSplit_4(seed = seed, numBatches = batches,logdir=None,vf_type='linear', animate=False)
    last4History.append(Rewards4)

    Rewards5, _, _ = cartpoleVpg(seed = seed, numBatches = batches,logdir=None,vf_type='linear', animate=False)
    allHistory.append(Rewards5)

MeanLast4= np.mean(last4History, axis = 0)
MeanAll = np.mean(allHistory, axis =0)
rewardPlot(MeanLast4, MeanAll)








