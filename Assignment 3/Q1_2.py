import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from astroML.resample import bootstrap
from astroML.stats import median_sigmaG

np.random.seed(100)
data = 1000  # Number of samples 
data_bootstrap = 10000  # Number of Bootstrap Samples   
dataset = stats.norm(0,1).rvs(data) # Gaussian samples = 1000

sample_median, user_stat = bootstrap(dataset, data_bootstrap, median_sigmaG, kwargs= {'axis' : 1})

mean_bootstrap_med = np.mean(sample_median)
std_dev_bootstrap_med = np.sqrt(np.pi/(2*data)) 

dist = stats.norm(mean_bootstrap_med, std_dev_bootstrap_med)
x = np.linspace(-0.4, 0.4, 2000)

plt.figure(figsize = (10,7))
plt.plot(x, dist.pdf(x), 'r--', label = 'Gaussian distribution')
plt.hist(sample_median, bins = 15, density = True, histtype='step', color = 'c', label = "Median of Bootstrap Samples")
plt.title('Bootstrapping', color='Green', fontsize='15')
plt.xlabel('Median (m)--->', color='blue', fontsize=12)
plt.ylabel('p(m/x, I)--->', color='blue', fontsize=12)
plt.legend()
plt.show()