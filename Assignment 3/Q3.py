from scipy import stats
import numpy as np

chi_sq_dof = np.array([0.96, 0.24, 3.84, 2.85])
chi_sq_vals = chi_sq_dof*49  # Degrees of Freedom = 49

for x in chi_sq_vals:
 p_val = 1 - stats.chi2(49).cdf(x)
 print('P_value for chi square value of {0} is {1}'.format(x, p_val))