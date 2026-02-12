import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define exponential function: y = a * exp(b * x)

def power_law_func(x, a, b):
   return a * np.power(x, b)

#def exponential_func(x, a, b):
#    return a * np.exp(b * x)

# Load data from CSV file

csv_file = 'sample_data.csv'  # Change this to your CSV filename
data = pd.read_csv(csv_file)

# Extract x and y data
x_data = data['x'].values  
y_data = data['y'].values  

# Perform exponential fit
# Initial guess for parameters [a, b]
initial_guess = [1.0, 0.1]
params, covariance = curve_fit(power_law_func, x_data, y_data, p0=initial_guess)

# Extract fitted parameters
a_fit, b_fit = params
print(f"Fitted parameters:")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"Power law fit: y = {a_fit:.4f} * exp({b_fit:.4f} * x)")


# Plot the data and fit
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='blue', alpha=0.6)

# Generate smooth curve for the fit
x_fit = np.linspace(x_data.min(), x_data.max(), 200)
y_fit = power_law_func(x_fit, a_fit, b_fit)
plt.plot(x_fit, y_fit, label=f'Fit: y = {a_fit:.2f}*x^({b_fit:.2f}*x)', 
         color='red', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_fit_power_law.png', dpi=300)
plt.show()

print("\nPlot saved as 'data_fit_power_law.png'")