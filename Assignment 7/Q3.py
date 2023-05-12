import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

df = pd.read_csv("c:/Users/Guguloth Abhishek/OneDrive/Desktop/DSA course/Assignment 7/q3_file.csv")
x = np.linspace(-0.5, 5.5, 1000)

# instantiate and fit the KDE model
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(df[: , np.newaxis])
gauss_dist = np.exp(kde.score_samples(x[:, np.newaxis]))

kde1 = KernelDensity(kernel='exponential', bandwidth=0.2).fit(df[: , np.newaxis])
exp_dist = np.exp(kde1.score_samples(x[:, np.newaxis]))

plt.plot(x, gauss_dist, 'v', label='Gaussian Distribution')
plt.plot(x, exp_dist, '--c', label = 'Exponential Distribution')
plt.title('KDE Estimate of the Quasar Redshift Distribution')
plt.legend()
plt.show()