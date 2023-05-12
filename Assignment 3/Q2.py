import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_values = np.array([203, 58, 210, 202, 198, 158, 165, 201, 157, 131, 166, 160, 186, 125, 218, 146])
y_values = np.array([495, 173, 479, 504, 510, 416, 393, 442, 317, 311, 400, 337, 423, 334, 533, 344])
y_value_errors = np.array([21, 15, 27, 14, 30, 16, 14, 25, 52, 16, 34, 31, 42, 26, 16, 22])

def line_func(x, m, b):
    return (m * x) + b

popt, pcov = curve_fit(line_func, x_values, y_values, sigma=y_value_errors, absolute_sigma=True, p0=[2.4, 40])
m_opt, b_opt = popt

print('Optimum slope (m) = {0}'.format(m_opt))
print('Optimum intercept (b) = {0}'.format(b_opt))

x_model = np.linspace(0, 300, 50)
y_model = line_func(x_model, m_opt, b_opt)

plt.scatter(x_values, y_values, color='red', s=7, label='Data points')
plt.errorbar(x_values, y_values, y_value_errors, fmt=' ', ecolor='black', elinewidth=1, capsize=3, capthick=1, label='Uncertainities')
plt.plot(x_model, y_model, color='violet', label='Best-fit plot')
plt.title('Chi Square Minimization best-fit curve', color='blue', fontsize=14)
plt.xlabel('x--->', fontsize=16, color='green')
plt.ylabel('y--->', fontsize=16, color='green')

leg = plt.legend(loc='upper left', fontsize=10.5)
for text in leg.get_texts():
 text.set_color("black")

plt.show()