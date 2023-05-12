import numpy as np
from scipy import stats
from scipy.stats import pearsonr

mean = 0
std_dev = 1
x = np.random.normal(mean, std_dev, 1000)
y = np.random.normal(mean, std_dev, 1000)

# Pearson coefficient and P-value
pear_cf, pear_p = pearsonr(x,y)
print('Pearson Correlation Coefficient value using scipy:',pear_cf )
print('Pearson P-Value using scipy: ', pear_p)

# P-value using Student-T distribution 
# Degrees of Freedom = 1000-2 = 998
t = pear_cf*np.sqrt(998/(1-(pear_cf**2))) 
student_t_pval = 2*(1-stats.t.cdf(t, 998))
print('P-value using Student-T distribution: ', student_t_pval)

if(np.round(pear_p, 5) == np.round(student_t_pval, 5)):
    print("Pearson P-value agrees with the Student-T P-value")