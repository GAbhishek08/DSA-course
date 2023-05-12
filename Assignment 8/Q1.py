import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroML.correlation import bootstrap_two_point_angular

df = pd.read_csv('C:\\Users\\Guguloth Abhishek\\OneDrive\\Desktop\\DSA course\\Assignment 8\\Q1.csv')
df = df[(df['r-mag']>17) & (df['r-mag']<20) & (df['spread_model']>0.002)]

np.random.seed(0)
bins = 10 ** np.linspace(np.log10(1. / 60.), np.log10(6), 16)
corr, err_corr, bootstrap = bootstrap_two_point_angular(df['#RA'],df['DEC'],
    bins=bins,method='landy-szalay',Nbootstraps=10)

plt.figure(figsize=(10, 7))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plt.errorbar(bin_centers, corr, err_corr, fmt='.c', ecolor='black', lw=1)
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Theta(degrees)--->', color='blue')
plt.ylabel(r'$\hat{w}(\theta)$--->', color='blue')
plt.title('Two-point Angular Correlation', color='red')
plt.show()