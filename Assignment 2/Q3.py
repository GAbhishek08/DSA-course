from scipy import stats
from scipy.stats import dweibull
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Weibull.csv')
x = df['class']
y = df['frequency in %']
plt.step(x, y, color='violet', label='Probability Distribution')
plt.xlim(0, 20)
plt.ylim(0, 16)

dist = stats.weibull_min(2,0,6)
t = np.arange(0, 21, 1)
s = 100*dist.pdf(t) 
plt.plot(t, s, color='red', label='Weibull Plot')

plt.xticks(np.arange(0, 21, 2), np.arange(0, 21, 2))
plt.xlabel('Wind speed (m/s) ----->', color='blue', fontsize=12)
plt.ylabel('Frequency (%) ----->', color='blue', fontsize=12)
plt.title("Wind Speed(m/s) - Frequency (%)", color='blue', fontsize=15)
plt.legend(loc='upper right')

plt.show()