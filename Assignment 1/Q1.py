import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Part 1
x=np.linspace(-10, 10, 1000)

mean1 = 1.5
std_dev = 0.5
dist = stats.norm(mean1, std_dev)

plt.plot(x, dist.pdf(x))
y=dist.pdf(x)

plt.title('Normal distribution')
plt.xlabel('Random variable(x)')
plt.ylabel('Probability')
plt.show()


# Part 2

# Mean calculation
mean2 = np.mean(y)
print(mean2)

# Median calculation
median = np.median(y)
print(median)

# Skewness calculation
skew=stats.skew(y)
print(skew)

# Kurtosis calculation
kurto=stats.kurtosis(y)
print(kurto)

# standard deviation
mad = stats.median_abs_deviation(y)
new_std=1.482*mad
print(new_std)

# sigma G calculation
q_25, q_75 = np.percentile(y,[25,75])
sigma = 0.7413 * (q_75 - q_25)
print(sigma)