import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


f = pd.read_csv('exoplanet.eu_catalog.csv')
f = f['eccentricity']

f_2 = f.dropna()
f_2.drop(f_2[f_2 == 0].index, inplace = True)

plt.hist(f_2, bins = 50, color = 'green', alpha = 0.5)
plt.xlim(xmin = 0, xmax = 1)
plt.ylabel('Exoplanets')
plt.xlabel('Eccentricity')
plt.show()