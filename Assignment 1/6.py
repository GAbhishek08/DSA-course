import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

f=pd.read_csv('exoplanet.csv')
f=f['eccentricity']
f_2=f.dropna()
f_2.drop(f_2[f_2==0].index, inplace=True)
e_new=stats.boxcox(f_2)
plt.hist(e_new, bins=20, alpha=0.5)

plt.ylabel('no of exoplanets')
plt.xlabel('eccentricity')
plt.show()