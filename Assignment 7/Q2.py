import numpy as np
import nestle

Data= np.array([[0.42, 0.72, 0., 0.3, 0.15, 0.09, 0.19, 0.35, 0.4, 0.54, 0.42, 0.69, 9.2, 0]
                [0.33, 0.41, -0.22, 0.01, -0.05, -0.05, -0.12, 0.26, 0.29, 0.39, 0.31, 0.42, -0]
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

def poly_fit(theta, x):
    return sum(t * x ** n for (n, t) in enumerate(theta))

def log_likelihood(theta, data=Data):
    x, y, y_err = data
    yM= poly_fit(theta, x)
    return -0.5 * np.sum(np.log(2 * np.pi * y_err ** 2) + (y - yM) ** 2 / y_err ** 2)

def log_prior(theta):
    return 200 * theta - 100

np.random.seed(1)
linear = nestle.sample(log_likelihood, log_prior, 2)
print(f"Log-evidence value for linear Model: {linear.logz}")

quad = nestle.sample(log_likelihood, log_prior, 3)
print(f"Log-evidence value for Quadratic Model: {quad. logz}")