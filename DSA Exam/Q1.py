import math

# Value of H_0 in one experiment in km/sec/Mpc
H1 = 67
sigma1 = 2.3

# Value of H_0 in another experiment in km/sec/Mpc
H2 = 71
sigma2 = 1.3

# Calculating the significance of the discrepancy between the two measurements
sigma = abs(H1 - H2) / math.sqrt(sigma1**2 + sigma2**2)

print(f"The significance of the discrepancy between the two measurements is {sigma:.2f} sigmas.")