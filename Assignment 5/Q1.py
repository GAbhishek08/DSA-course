from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

densities = np.array([2.12, 2.71, 3.44, 2.76, 2.72, 0.96, 2.00, 3.26, 2.50, 1.20, 1.62, 1.30, 1.96, 
2.60, 1.30, 2.67, 4.40, 1.80, 4.90, 2.39, 1.62, 1.47, 0.89, 2.52, 1.21, 0.90, 0.80])

log_densities = np.log(densities)
shapiro_test_densities = stats.shapiro(densities)
shapiro_test_log_densities = stats.shapiro(log_densities)

print("W(Shapiro Test Statistic) value for densities:", shapiro_test_densities.statistic)
print("P-Value from Shapiro Test for densities:", shapiro_test_densities.pvalue)
print("W(Shapiro Test Statistic) value for log(densities):", shapiro_test_log_densities.statistic)
print("P-Value from Shapiro Test for log(densities):", shapiro_test_log_densities.pvalue)

array = [shapiro_test_densities.pvalue, shapiro_test_log_densities.pvalue ]

if array[0] > 0.05:
    print("NUll Hypothesis for densities is accepted since the values are taken from a population that follows Normal Distribution")
else:
    print("NUll Hypothesis for densities is rejected since the values are taken from a population that does not follow Normal Distribution")

if array[1] > 0.05:
    print("NUll Hypothesis for log(densities) is accepted since the values are taken from a population that follows Normal Distribution")
else:
    print("NUll Hypothesis for log(densities) is rejected since the values are taken from a population that does not follow Normal Distribution")

print('Since the p-value of log(densities) > densities, the data set "log_densities" is more closer to Gaussian')

mean, std_dev = stats.norm.fit(densities)
mean1, std_dev1 = stats.norm.fit(log_densities)
x1 = np.linspace(-8, 8, 70)
x2 = np.linspace(-8, 8, 70)
density_fit = stats.norm.pdf(x1, mean, std_dev)
log_density_fit = stats.norm.pdf(x2, mean1, std_dev1)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Normally fit Histograms', color='red', fontsize =15)
ax1.plot(x1, density_fit, color='black', label="Guassian fit_Densities")
ax1.hist(densities, bins='fd', color = 'green', label='Densities', density=True )
ax1.set_title('Densities', color='red', fontsize=12)
ax1.set_xlabel('Densities--->', color='blue', fontsize=10)
ax1.set_ylabel('Frequency--->', color='blue', fontsize=10)
ax1.set_xlim([-5, 6])
ax1.legend()

ax2.plot(x2, log_density_fit, color='black', label="Guassian fit_Log(Densities)")
ax2.hist(log_densities, bins='fd', color = 'green' , label='Log_densities', density=True)
ax2.set_title('Log_Densities', color='red', fontsize=12)
ax2.set_xlabel('Log(Densities)--->', color='blue', fontsize=10)
ax2.legend()
ax2.set_xlim([-5, 2.5])
plt.show()

print('The histograms of densities and log(densities) also verify that the data set "log_densities" is more closer to the Guassian' )