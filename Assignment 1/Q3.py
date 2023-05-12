import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


std_dev = 5**0.5
mean = 5

poi_dist = stats.poisson(mean)
gaus_dist = stats.norm(mean,std_dev)

x = np.linspace(-10,10,1000)
plt.plot(x, poi_dist.pmf(x), ls='-', color='red', label='poisson')
plt.plot(x, gaus_dist.pdf(x), ls='--', color='green', label='gaussian')
plt.legend()
plt.show()