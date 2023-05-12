import math
from math import log
import numpy as np
from scipy import stats
from scipy import optimize
import random

random.seed(123)

#source code
data = np.array([[ 0.42,  0.72,  0.  ,  0.3 ,  0.15,
                   0.09,  0.19,  0.35,  0.4 ,  0.54,
                   0.42,  0.69,  0.2 ,  0.88,  0.03,
                   0.67,  0.42,  0.56,  0.14,  0.2  ],
                 [ 0.33,  0.41, -0.22,  0.01, -0.05,
                  -0.05, -0.12,  0.26,  0.29,  0.39, 
                   0.31,  0.42, -0.01,  0.58, -0.2 ,
                   0.52,  0.15,  0.32, -0.13, -0.09 ],
                 [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,
                   0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1  ]])

x, y, sigma_y = data

# To calculate Naive likelihood values
def polynomial_fit(theta, x):
    """Polynomial model of degree (len(theta) - 1)"""
    return sum(t * x ** n for (n, t) in enumerate(theta))

def log_L(theta, model = polynomial_fit, data = data):
    """Gaussian log-likelihood of the model at theta"""
    x, y, sigma_y = data
    y_fit = model(theta, x)
    return sum(stats.norm.logpdf(*args)
               for args in zip(y, y_fit, sigma_y))

def best_theta(degree, model = polynomial_fit, data = data):
    theta_0 = (degree + 1) * [0]
    neg_logL = lambda theta: -log_L(theta, model, data)
    return optimize.fmin_bfgs(neg_logL, theta_0, disp = False)

theta1 = best_theta(1)
theta2 = best_theta(2)


# This code, AIC uses a model's maximum likelihood estimation (log-likelihood) as a measure of fit
linear_L = log_L(best_theta(1))
Quad_L = log_L(best_theta(2))
k = 3  # No.of free parameters for quadratic model
k1 = 2  # No. of free parameters for linear model
N = 20 # Number of data points

Aic_Linear = -2*(log(linear_L)) + 2*k1
Bic_Linear = -2*(log(linear_L)) + k1*(log(N))

Aic_Quad = -2*(log(Quad_L)) + 2*k
Bic_Quad = -2*(log(Quad_L)) + k*(log(N))

print("AIC for Linear Model = {0}".format(Aic_Linear))
print("BIC for Linear Model = {0}".format(Bic_Linear))
print("AIC for Quadratic Model = {0}".format(Aic_Quad))
print("BIC for Quadratic Model = {0}".format(Bic_Quad))
print("AIC and BIC values of Linear model are less than the quadratic model")
print('Hence, Linear model is the best-fit one ')
print("Yes, the results agree with the frequentist model comparison results shown on the blog")

AIC = np.array([Aic_Linear, Aic_Quad])
BIC = np.array([Bic_Linear, Bic_Quad])

d_AIC = np.sum(AIC) - np.min(AIC)
d_BIC = np.sum(BIC) - np.min(BIC)

if 0<=d_AIC<=2:
    print("The level of Empirical Support For Model is Substantial")

if 4<=d_AIC<=7:
    print("The level of Empirical Support For Model is Considerably Less")

if d_AIC>10:
    print("The level of Empirical Support For Model is Essentially none")

if 0<=d_BIC<2:
   print("Evidence against the Model is Not Worth More Than A Bare Mention") 

if 2<=d_BIC<=6:
   print("Evidence against the Model is Positive")

if 6<=d_BIC<=10:
   print("Evidence against the Model is Strong")

if d_BIC>10:
   print("Evidence against the Model is Very Strong")
