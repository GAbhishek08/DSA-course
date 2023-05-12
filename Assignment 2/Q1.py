import numpy as np
import matplotlib.pyplot as plt
import random

s = np.random.chisquare(3, 1000000)

# For N = 1, we have:
samples = [np.mean(random.choices(s, k = 1)) for _ in range(1000000)]
plt.subplot(3, 1, 1)
plt.hist(samples, bins = 100, color = 'blue', label = 'N = 1')
plt.ylabel('p(x)', color = 'red', fontsize = 10)
plt.legend(loc = 'upper right')
plt.xlim(0, 10)

# For N = 5, we have:
samples1 = [np.mean(random.choices(s, k = 5)) for _ in range(1000000)]
plt.subplot(3, 1, 2)
plt.hist(samples1, bins = 100, color = 'blue', label = 'N = 5')
plt.ylabel('p(x)', color = 'red', fontsize = 10)
plt.legend(loc = 'upper right')
plt.xlim(0, 10)

# For N = 10, we have:
samples2 = [np.mean(random.choices(s, k = 10)) for _ in range(1000000)]
plt.subplot(3, 1, 3)
plt.hist(samples2, bins = 100, color = 'blue', label = 'N = 10')
plt.xlabel('x --->', color = 'red', fontsize = 10)
plt.ylabel('p(x)', color = 'red', fontsize = 10)
plt.legend(loc = 'upper right')
plt.xlim(0, 10)
plt.show()