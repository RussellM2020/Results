import matplotlib.pyplot as plt
from last1_Vpg import cartpoleSplit_1
from last2_Vpg import cartpoleSplit_2
from last3_Vpg import cartpoleSplit_3
from last4_Vpg import cartpoleSplit_4
from dualCopy import cartpoleVpg


batches = 75
Iterations= range(0,batches) 

Rewards1, SL_loss1, PG_loss1 = cartpoleSplit_1(numBatches = batches,logdir=None,vf_type='linear', animate=False) # when you want to start collecting results, set the logdir
Rewards2, SL_loss2, PG_loss2 = cartpoleSplit_2(numBatches = batches,logdir=None,vf_type='linear', animate=False)
Rewards3, SL_loss3, PG_loss3 = cartpoleSplit_3(numBatches = batches,logdir=None,vf_type='linear', animate=False)
Rewards4, SL_loss4, PG_loss4 = cartpoleSplit_4(numBatches = batches,logdir=None,vf_type='linear', animate=False)
Rewards5, _, PG_loss5 = cartpoleVpg(numBatches = batches,logdir=None,vf_type='linear', animate=False)



def rewardPlot():
    plt.xlabel("Iterations")
    plt.ylabel("Sum of Rewards")
    
    plt.plot(Iterations, Rewards1, '-r', label  = 'Last 1 layer vpg')
    plt.plot(Iterations, Rewards2, '-b', label  = 'Last 2 layers vpg')
    plt.plot(Iterations, Rewards3, '-g', label  = 'Last 3 layers vpg')
    plt.plot(Iterations, Rewards4, '-c', label  = 'Last 4 layers vpg')
    plt.plot(Iterations, Rewards5, '-k', label  = 'all 5 layers vpg')


    plt.axis([0, batches,0, 202])
    plt.title("Reward Comparison by Layer of Split")
    plt.legend()
    
    plt.savefig("reward_split_comp.png")

def slPlot():
    plt.clf()
    plt.xlabel("Iterations")
    plt.ylabel("Loss under Supervised learning")
    
    plt.plot(Iterations, SL_loss1, '-r', label  = 'Last 1 layer vpg')
    plt.plot(Iterations, SL_loss2, '-b', label  = 'Last 2 layers vpg')
    plt.plot(Iterations, SL_loss3, '-g', label  = 'Last 3 layers vpg')
    plt.plot(Iterations, SL_loss4, '-c', label  = 'Last 4 layers vpg')


    plt.axis([0, batches,0, 15])
    plt.title("SL loss Comparison by Layer of Split")
    plt.legend()
    
    plt.savefig("SLloss_split_comp.png")

def pgPlot():
    plt.clf()
    plt.xlabel("Iterations")
    plt.ylabel("Loss under Policy Gradient")
    
    plt.plot(Iterations, PG_loss1, '-r', label  = 'Last 1 layer vpg')
    plt.plot(Iterations, PG_loss2, '-b', label  = 'Last 2 layers vpg')
    plt.plot(Iterations, PG_loss3, '-g', label  = 'Last 3 layers vpg')
    plt.plot(Iterations, PG_loss4, '-c', label  = 'Last 4 layers vpg')
    plt.plot(Iterations, PG_loss5, '-k', label  = 'all 5 layers vpg')


    plt.axis([0, batches,-1, 0.1])
    plt.title("PG loss Comparison by Layer of Split")
    plt.legend()
    
    plt.savefig("pgloss_split_comp.png")

rewardPlot()
slPlot()
pgPlot()