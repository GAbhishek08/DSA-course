import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

np.random.seed(100)
data = np.random.normal(0, 1, size=1000)

sample_median = [] 
for i in range (10000):
    x = random.sample(data.tolist(), 386)
    med = np.median(x)
    sample_median.append(med)

mean_bootstrap_med = np.mean(sample_median)
std_dev_bootstrap_med = math.sqrt(np.pi/(2*1000))

dist = stats.norm(mean_bootstrap_med, std_dev_bootstrap_med)
x = np.linspace(-0.4, 0.4, 2000)
p = dist.pdf(x)

plt.figure(figsize=(10,7))
plt.plot(x, p , 'r--', label='Gaussian Distribution')
plt.hist(sample_median, bins=15, density=True, histtype='step', color='c', label='Median of Bootstrap Samples')
plt.title('Bootstrapping', color='Green', fontsize='15')
plt.xlabel('Median (m)--->', color='blue', fontsize=12)
plt.ylabel('p(m/x, I)--->', color='blue', fontsize=12)
plt.legend()
plt.show()