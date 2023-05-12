import pandas as pd
from math import log
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

df = pd.read_csv('test_file.csv')
plt.scatter(df['red'], df['lum'], color='Purple')
plt.xscale("log")
plt.yscale("log")
plt.xlabel('Redshift', fontsize=12, color='red')
plt.ylabel('Luminosity', fontsize=12, color='red')
plt.title("Luminosity-Redshift", color='red', fontsize=15)
plt.show()
# Data sets are "Positively Correlated"

# Calculating correlation coefficients and corresponding p values
sp_cf_p = spearmanr(df['red'], df['lum'])
pear_cf_p = pearsonr(df['red'], df['lum'])
kend_cf_p = kendalltau(df['red'], df['lum'])

print('Spearman Correlation  = ', sp_cf_p)
print("Pearson Correlation  = ", pear_cf_p)
print("Kendall Tau Correlation  = ", kend_cf_p)

#Calculating p-value of null hypothesis
t_val, p_val = ttest_ind(df['red'], df['lum']) 
print('P-value for Null Hypothesis using ttest = ', p_val)