from scipy import stats
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

ndim = 3
nburn_period = 100
nsteps = 2000
nwalkers = 50

x_data = np.array([203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y_data = np.array([495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
y_data_err = np.array([21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])

def post_log(theta, x, y, sig):
    a, b, sig = theta

    if(sig>=0):
        prior=-(np.log(sig)+1.5*np.log(1+pow(b,2)))
    else: 
        prior = -np.inf

    y_model = a+b*x
    sigma_model = pow(sig, 2) + np.exp(2*sig)*pow(y_model, 2)
    likelihood = -0.5*np.sum((y-y_model)**2/sigma_model +np.log(sigma_model))
    return prior+likelihood

np.random.seed(123)
initial_guess = np.random.random((nwalkers, ndim))
sampler = emcee.EnsembleSampler(nwalkers, ndim, post_log, args=[x_data, y_data, y_data_err])
sampler.run_mcmc(initial_guess, nsteps)
chain = sampler.get_chain(discard=nburn_period, thin=20, flat=True)
emcee_trace = sampler.chain[: nburn_period:, :].reshape(-1, ndim).T
fig = corner.corner(chain, levels = (0.68, 0.95), labels=["a(slope)", "b(intercept)", "sig"]);
fig.suptitle('68% & 95% Confidence Intervals for a and b')
plt.show()