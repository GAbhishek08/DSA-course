import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import nestle
import emcee
from astroML.plotting import plot_mcmc

data = pd.read_csv("c:/Users/Guguloth Abhishek/OneDrive/Desktop/DSA course/Assignment 7/q1_file_1.csv")
z = data['z']
fgas = data['fgas']
fgas_err = data['fgas_error']

def log_prior(theta):
    f0, f1, sigma = theta
    if sigma > 0 and (0 < f0 < 0.5) and (-0.5 < f1 < 0.5):
        return -1.5 * np.log(1 + (f0 * f1) ** 2) - np.log(sigma)
    else:
        return -np.inf

def log_likelihood(theta, x, y):
    f0, f1, sigma = theta
    y_model = f0 + f0 * f1 * x
    return -0.5 * np.sum(np.log(2 * np.pi * sigma * 2) + (y - y_model) * 2 / sigma ** 2)

def log_posterior(theta, x, y):
    return log_prior(theta) + log_likelihood(theta, x, y)

#MCMC parameters
ndim = 3
nburn = 1000
nsteps = 10000
nwalkers = 50

#Initial guesses
initial_guess = np.random.random((nwalkers, ndim))

#begin MCMC
sampler = emcee. EnsembleSampler(nwalkers, ndim, log_posterior, args=[z, fgas, fgas_err])
sampler.run_mcmc(initial_guess, nsteps, progress = True)
trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T[:2]

#plot
best_theta_bayes = np.mean(sampler[:, :2], 0)
print(f"The estimated value of the parameter f0 is - {best_theta_bayes[0]}")
print(f"The estimated value of the parameter fl is - {best_theta_bayes[1]}")

fig = plt.figure()
ax = plot_mcmc(trace, fig=fig, limits = [(-1, 1), (-0.25, 0.25)], levels = [0.68, 0.90], labels=["f0"]) 
plt.title("68% & 90% Credible Intervals of f0 and f1 parameters", color = 'red')
plt.show()