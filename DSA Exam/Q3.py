import numpy as np

# Define the reduced chi-square values and number of free parameters for each model
x2_a = 1.3
k_a = 6
x2_b = 0.7
k_b = 8

# Define the number of data points
n = 10

# Calculate the likelihoods for each model
L_a = np.exp(-0.5 * x2_a * n)
L_b = np.exp(-0.5 * x2_b * n)

# Calculate the BIC for each model
BIC_a = k_a * np.log(n) - 2 * np.log(L_a)
BIC_b = k_b * np.log(n) - 2 * np.log(L_b)

print("BIC for Model A: {:.2f}".format(BIC_a))
print("BIC for Model B: {:.2f}".format(BIC_b))