
from SplitImage import SplitImage
from VPGImage_5 import VPGImage_5
from VPGImage_2 import VPGImage_2
from VPGImage_5Conv import VPGImage_5Conv


batches = 1

general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=4000, n_iter=batches, initial_stepsize=1e-3)

RewardSplit, PGSplit = SplitImage(logdir=None, seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, use_adaptive_stepsize=False, split_level=0, sup_learning_steps = 1, **general_params)
RewardV5, PGV5 = VPGImage_5(logdir=None, seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, use_adaptive_stepsize=False, split_level=0, sup_learning_steps = 1, **general_params)
RewardV2, PGV2 = VPGImage_2(logdir=None, seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, use_adaptive_stepsize=False, split_level=0, sup_learning_steps = 1, **general_params)
RewardV5_Conv, PGV5_Conv = VPGImage_5Conv(logdir=None, seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, use_adaptive_stepsize=False, split_level=0, sup_learning_steps = 1, **general_params)


Iterations = Iterations= range(0,numBatches) 
 

def rewardPlot():
    plt.xlabel("Iterations")
    plt.ylabel("Sum of Rewards")
    
    plt.plot(Iterations, RewardSplit, '-r', label  = '1 Conv Block SL, 2 FCC vpg')
    plt.plot(Iteration, RewardV5_Conv, '-k', label = '1 Conv Block vpg, 2 FCC vpg')
    plt.plot(Iterations, RewardV5, '-b', label  = '5 FCC vpg')
    plt.plot(Iterations, RewardV2, '-g', label  = '2 FCC vpg')
    

    plt.axis([0, batches,0, 202])
    plt.title("Reward Comparison")
    plt.legend()
    
    plt.savefig("Split_imgReward_1.png")

def pgPlot():
    plt.xlabel("Iterations")
    plt.ylabel("PG Loss")
    
    plt.plot(Iterations, PGSplit, '-r', label  = '1 Conv Block SL, 2 FCC vpg')
    plt.plot(Iteration, PGV5_Conv, '-k', label = '1 Conv Block vpg, 2 FCC vpg')
    plt.plot(Iterations, PGV5, '-b', label  = '5 FCC vpg')
    plt.plot(Iterations, PGV2, '-g', label  = '2 FCC vpg')
    

    plt.axis([0, batches,-1.5, 0])
    plt.title("PG Loss Comparison")
    plt.legend()
    
    plt.savefig("Split_imgPGLoss_1.png")