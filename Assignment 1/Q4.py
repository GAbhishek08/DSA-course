import numpy as np


lifetime = [0.8920, 0.881, 0.8913, 0.9837, 0.8958]
error = [0.00044, 0.009, 0.00032, 0.00048, 0.00045]

sum1 = 0
for i in range(5):
    sum1 += (lifetime[i] / ((error[i])**2))

sum2 = 0
for i in range(5):
    sum2 += (1 / ((error[i])**2))

avg_mean = sum1 / sum2
print(avg_mean)

error_mean = (1 / sum2)**0.5
print(error_mean)