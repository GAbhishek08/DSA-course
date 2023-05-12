import scipy.stats as stats

k = 2 # degrees of freedom
N = 10 # number of data points

confidence_level = 0.85 # 85 percent


chi_squared = stats.chi2.ppf(confidence_level, k) # calculate the chi-squared value corresponding to 85% confidence level
delta_chi_squared = [1, 2.3, 4.61]

# calculate the corresponding values of delta X^2
delta_x_squared = [(delta_chi_squared[i] / k) * chi_squared for i in range(len(delta_chi_squared))]

print("Delta X^2 values for 85% confidence intervals:")
print(delta_x_squared)