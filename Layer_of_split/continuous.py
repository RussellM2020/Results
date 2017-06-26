from vpg_continuous import main_pendulum


for seed in range(0,100,20):
    Rewards4,_,_= cartpoleSplit_4(seed = seed, numBatches = batches,logdir=None,vf_type='linear', animate=False)
    last4History.append(Rewards4)

    Rewards5, _, _ = cartpoleVpg(seed = seed, numBatches = batches,logdir=None,vf_type='linear', animate=False)
    allHistory.append(Rewards5)