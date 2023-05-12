import pandas as pd
import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
import math
from math import log
from scipy import stats
import random
import matplotlib.pyplot as plt

random.seed(123)

# Reading the data
df = pd.read_csv('C:\Users\Guguloth Abhishek\OneDrive\Desktop\DSA course\Assignment 4\data_science_A4_Q1.csv')
data = np.array([df.x_data, df.y_data, df.sigma_y])
x_data, y_data, sigma_y = data

# Define a linear model function
def linear_func(x, a, b):
    return a*x + b

# Define a quadratic model function
def quad_func(x, a, b, c):
    return a*(x**2) + b*x + c

# Define a cubic model function
def cubic_func(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d

# Optimum Parameters for the Linear Model
opt_lin, cov_lin = curve_fit(linear_func, x_data, y_data, sigma=sigma_y, absolute_sigma=True)
a_lin, b_lin = opt_lin

# Optimum parameters for the quadratic model
opt_quad, cov_quad = curve_fit(quad_func, x_data, y_data, sigma=sigma_y, absolute_sigma=True)
a_quad, b_quad, c_quad = opt_quad

# Optimum parameters for the cubic model
opt_cubic, cov_cubic = curve_fit(cubic_func, x_data, y_data, sigma=sigma_y, absolute_sigma=True)
a_cub, b_cub, c_cub, d_cub = opt_cubic

print("Best Fit values for the Linear Model:\n a = {0}, b = {1}".format(a_lin, b_lin))
print("Best fit values for the Quadratic Model: \n a = {0}, b = {1}, c = {2}".format(a_quad, b_quad, c_quad))
print("Best fit values for the Cubic Model: \n a = {0}, b = {1}, c = {2}, d = {3}".format(a_cub, b_cub, c_cub, d_cub))

# Frequentist Model Comparision to find the best fit model
def polynomial_fit(theta, x):
    """Polynomial model of degree (len(theta) - 1)"""
    return sum(t * x ** n for (n, t) in enumerate(theta))

def logL(theta, model=polynomial_fit, data=data):
    """Gaussian log-likelihood of the model at theta"""
    x_data, y_data, sigma_y = data
    y_fit = model(theta, x_data)
    return sum(stats.norm.logpdf(*args)
               for args in zip(y_data, y_fit, sigma_y))

def best_theta(degree, model=polynomial_fit, data=data):
    theta_0 = (degree + 1) * [0]
    neg_logL = lambda theta: -logL(theta, model, data)
    return optimize.fmin_bfgs(neg_logL, theta_0, disp=False)

def compute_chi2(degree, data=data):
    x_data, y_data, sigma_y = data
    theta = best_theta(degree, data=data)
    resid = (y_data - polynomial_fit(theta, x_data)) / sigma_y
    return np.sum(resid ** 2)

def compute_dof(degree, data=data):
    return data.shape[1] - (degree + 1)

def chi2_likelihood(degree, data=data):
    chi2 = compute_chi2(degree, data)
    dof = compute_dof(degree, data)
    return stats.chi2(dof).pdf(chi2)

print("Chi2 Likelihood value of Linear Model:", chi2_likelihood(1))
print("Chi2 Likelihood value of Quadratic Model: ", chi2_likelihood(2))
print("Chi2 Likelihood value of Cubic Model:", chi2_likelihood(3))
max_likelihood = max(chi2_likelihood(1), chi2_likelihood(2), chi2_likelihood(3))
print("The maximum Chi2 Likelihood = {0} and cooresponds to the Linear Model".format(max_likelihood))
print("Linear Model fits the data best since it has the highest Chi2 Likelihood")

# AIC and BIC to find the best fit model
m_linear = 2 # Number of free parameters of linear model
m_quad = 3 # Number of free parameters of Quadratic model
m_cubic = 4 # Number of free parameters of Cubic model
N = 20 # Number of data points

linear_L = logL(best_theta(1))
Quad_L = logL(best_theta(2))
Cubic_L = logL(best_theta(3))

Aic_Linear = -2*(linear_L) + 2*m_linear
Bic_Linear = -2*(linear_L) + m_linear*(log(N))

Aic_Quad = -2*(Quad_L) + 2*m_quad
Bic_Quad = -2*(Quad_L) + m_quad*(log(N))

Aic_Cubic = -2*(Cubic_L) + 2*m_cubic
Bic_Cubic = -2*(Cubic_L) + m_cubic*(log(N))

print("AIC for Linear Model = {0}".format(Aic_Linear))
print("BIC for Linear Model = {0}".format(Bic_Linear))
print("AIC for Quadratic Model = {0}".format(Aic_Quad))
print("BIC for Quadratic Model = {0}".format(Bic_Quad))
print("AIC for Cubic Model = {0}".format(Aic_Cubic))
print("BIC for Cubic Model = {0}".format(Bic_Cubic))
print("AIC and BIC values of Linear model are less than the Quadratic and Cubic models")
print('Hence, Linear model is the best-fit one ')

# p-value when linear model is null hypothesis
pval_quad_lin = 1-stats.chi2(m_quad - m_linear).cdf(compute_chi2(1) - compute_chi2(2))
pval_cubic_lin = 1-stats.chi2(m_cubic - m_linear).cdf(compute_chi2(1) - compute_chi2(3))
print("p_value of Linear model compared with quadratic model: ", pval_quad_lin)
print("p-value of Linear Model compared with cubic model: ",pval_cubic_lin)

AIC = np.array([Aic_Linear, Aic_Quad, Aic_Cubic])
BIC = np.array([Bic_Linear, Bic_Quad, Bic_Cubic])

d_AIC = np.sum(AIC) - np.min(AIC)
d_BIC = np.sum(BIC) - np.min(BIC)

print("Delta AIC value for Linear MOdel: ", Aic_Linear-min(AIC) )
print("Delta AIC value for Quadratic MOdel: ", Aic_Quad-min(AIC) )
print("Delta AIC value for Cubic MOdel: ", Aic_Cubic-min(AIC) )

print("Delta BIC value for Linear MOdel: ", Bic_Linear-min(BIC) )
print("Delta BIC value for Quadratic MOdel: ", Bic_Quad-min(BIC) )
print("Delta BIC value for Cubic MOdel: ", Bic_Cubic-min(BIC) )

# Plotting
x = np.linspace(0,1,1000)
y_linear = linear_func(x, a_lin, b_lin)
y_quad = quad_func(x, a_quad, b_quad, c_quad)
y_cubic = cubic_func(x, a_cub, b_cub, c_cub, d_cub)
plt.errorbar(x_data, y_data, sigma_y, fmt='.', color='black', label = 'Data')
plt.plot(x, y_linear, color='violet', label='Linear Model')
plt.plot(x, y_quad, color='orange', label='Quadratic Model')
plt.plot(x, y_cubic, color='green', label='Cubic Model')
plt.xlabel('x--->', color='blue', fontsize=12)
plt.ylabel('y--->', color='blue', fontsize=12)
plt.title("Model Fitting and Comparision", color='blue', fontsize=15)
plt.legend()
plt.show()