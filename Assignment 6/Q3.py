import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import emcee

x_data = np.array([201, 244, 47, 287, 203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y_data = np.array([592, 401, 583, 402, 495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
y_data_err = np.array([61, 25, 38, 15, 21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])

# FREQUENTIST APPROACH (MAXIMUM LIKELIHOOD)
def huber_loss(t, c=3):
    return ((abs(t) >= c) * -c * (0.5 * c - abs(t)) + 0.5*pow(t, 2)*(abs(t) < c))

def total_huber_loss(theta, x=x_data, y=y_data, e=y_data_err, c=3):
    return huber_loss((y - theta[0] - theta[1] * x) / e, c).sum()

x = np.linspace(25, 350, 2000)
best_theta = optimize.fmin(total_huber_loss, [0, 0], disp=False)

plt.errorbar(x_data, y_data, y_data_err, fmt='.k', ecolor='black', label= "Data with errors")
plt.plot(x, best_theta[1]*x + best_theta[0], color='cyan', label = "Best Fit")
plt.title('Maximum Likelihood Fit: Huber Loss', color='red')
plt.xlabel('x--->', color='black')
plt.ylabel('y--->', color='black')
plt.legend()
plt.show()

# BAYESIAN APPROACH
def log_prior(theta):
    #g_i needs to be between 0 and 1
    if (all(theta[2:] > 0) and all(theta[2:] < 1)):
        return 0
    else:
        return -np.inf  # recall log(0) = -inf

def log_likelihood(theta, x, y, e, sigma_B):
    dy = y - theta[0] - theta[1] * x
    g = np.clip(theta[2:], 0, 1)  # g<0 or g>1 leads to NaNs in logarithm
    logL1 = np.log(g) - 0.5 * np.log(2 * np.pi * e * 2) - 0.5 * (dy / e) * 2
    logL2 = np.log(1 - g) - 0.5 * np.log(2 * np.pi * sigma_B * 2) - 0.5 * (dy / sigma_B) * 2
    return np.sum(np.logaddexp(logL1, logL2))

def log_posterior(theta, x, y, e, sigma_B):
    return log_prior(theta) + log_likelihood(theta, x, y, e, sigma_B)

ndim = 2 + len(x_data)  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nburn = 10000  # "burn-in" period to let chains stabilize
nsteps = 15000  # number of MCMC steps to take

# set theta near the maximum likelihood, with 
np.random.seed(0)
starting_guesses = np.zeros((nwalkers, ndim))
starting_guesses[:, :2] = np.random.normal(best_theta, 1, (nwalkers, 2))
starting_guesses[:, 2:] = np.random.normal(0.5, 0.1, (nwalkers, ndim - 2))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[x_data, y_data, y_data_err, 50])
sampler.run_mcmc(starting_guesses, nsteps, progress=True)

sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
new_theta = np.mean(sample[:,:2], 0)
g = np.mean(sample[:, 2:], 0)
outliers = (g < 0.40)

plt.errorbar(x_data, y_data, y_data_err, fmt='.k', ecolor='cyan', label = "Error bar")
plt.plot(x, new_theta[0] + new_theta[1] * x, color='black', label = "Bayesian Approach")
plt.plot(x_data[outliers], y_data[outliers], 'ro', ms=20, mfc='none', mec='red', label="outliers")
plt.legend()
plt.show()

# Plotting the fit obtained from the two approaches

plt.errorbar(x_data, y_data, y_data_err, fmt='.k', ecolor='cyan', label="Given Data")
plt.plot(x, best_theta[0]+best_theta[1]*x, color='blue', label='Frequentist(Maximum Likelihood) Approach')
plt.plot(x, new_theta[0]+new_theta[1]*x, color ='black', label='Bayesian Approach')
plt.legend()
plt.show()