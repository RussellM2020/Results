import sys
path = '/home/russellm/Research/Results/Split'
sys.path.append(path)
from fileHandling import read
#import matplotlib.pyplot as plt

_cur_ = '/SplitImage'



Rewards = read(path +_cur_+ "/exp_2/store1/VPGImage_5Conv_reward_store1.dat")

print(len(Rewards))
















