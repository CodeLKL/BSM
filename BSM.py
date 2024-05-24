import numpy as np
import matplotlib.pyplot as plt

# Parameters for Geometric Brownian Motion
S0 = 100     # Initial stock price
mu = 0.05    # Drift coefficient
sigma = 0.2  # Volatility coefficient
T = 1.0      # Time horizon (1 year)
dt = 0.01    # Time step
N = int(T / dt)  # Number of time steps

# Time vector
t = np.linspace(0, T, N)

# Brownian motion
W = np.random.standard_normal(size=N) 
W = np.cumsum(W) * np.sqrt(dt)  # Cumulative sum to simulate Brownian path

# Geometric Brownian Motion
S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)

# Plotting the result
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.title('Geometric Brownian Motion')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
