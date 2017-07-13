import numpy as np; 
#np.random.seed(22)
import seaborn as sns; sns.set(color_codes=True)

import matplotlib.pyplot as plt
x = np.linspace(0, 15, 31)
data = x + np.random.rand(10, 31) + np.random.randn(10, 1)


print(data)
# ax = sns.tsplot(data=data)
# plt.show()