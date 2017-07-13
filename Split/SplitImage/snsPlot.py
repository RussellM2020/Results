import sys
path = '/home/russellm/Research/Results/Split'
sys.path.append(path)
from fileHandling import read
from fileHandling import store
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(color_codes=True)
_cur_ = '/SplitImage/TRPO/trpoSplit30/rate_e-4'

rewStore = []
batches = 300

mean = read(path +_cur_+ "/DisRewMean.dat")
std = read(path +_cur_+ "/DisRewSTD.dat")

std = np.divide(std, mean)
std = np.absolute(np.multiply(std, 100*np.ones(batches)))
print(std)
print("*****")
print(mean)

Iterations = range(batches)


# fig = plt.figure()  # create a figure object
# ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
# ax.set_xlim(0,batches)
# ax.set_ylim(-2*10**4,-100)
# b, 'b-')
data = mean + np.random.random(300)
sanity = 20*np.ones(batches)
ax = sns.tsplot(data=data)

#ax.imshow(Mean, 'spring')
#ax.set_ylabel('some numbers')
plt.show()
#plt.savefig("trial.png")