import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


mua = 0
gamma = 1.5
sigma = 1.5

x = np.linspace(-10, 10, 1000)

cauchy_dist = stats.cauchy(mua, gamma)
gaus_dist = stats.norm(mua, sigma)

plt.plot(x, cauchy_dist.pdf(x), ls='-', color = 'green')
plt.plot(x, gaus_dist.pdf(x), ls='--', color = 'blue')
plt.legend(["cauchy distribution $\mu=0,\ \gamma=1.5$", "Guassion distribution $\mu=0,\ \sigma=1.5$"], loc ="upper right")
plt.show()