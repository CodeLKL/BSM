import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# Parameters for Geometric Brownian Motion
S0 = 100     # Initial stock price
mu = 0.05    # Drift coefficient
sigma = 0.2  # Volatility coefficient
T = 1.0      # Time horizon (1 year)
dt = 0.01    # Time step
N = int(T / dt)  # Number of time steps
num_simulations = 1000  # Number of simulations

# Time vector
t = np.linspace(0, T, N)

# Matrix to store the simulation results
simulations = np.zeros((num_simulations, N))

# Running the simulations
for i in range(num_simulations):
    # Brownian motion
    W = np.random.standard_normal(size=N) 
    W = np.cumsum(W) * np.sqrt(dt)  # Cumulative sum to simulate Brownian path
    
    # Geometric Brownian Motion
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    
    # Store the result
    simulations[i, :] = S

# Calculate the average of the endpoint values
endpoints = simulations[:, -1]
average_endpoint = np.mean(endpoints)

# Normalize the endpoints to the range [0, 1] for colormap
norm = mcolors.Normalize(vmin=min(endpoints), vmax=max(endpoints))

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 1]})

# Plot the GBM simulations on the first subplot with colormap
for i in range(num_simulations):
    color = plt.cm.coolwarm(norm(endpoints[i]))
    axs[0].plot(t, simulations[i, :], lw=0.5, alpha=0.6, color=color)
axs[0].set_title('Geometric Brownian Motion - 1000 Simulations')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Stock Price')
axs[0].grid(True)

# Plot the distribution of the endpoint values on the second subplot
sns.histplot(endpoints, bins=50, kde=True, color='blue', ax=axs[1])
axs[1].axvline(average_endpoint, color='red', linestyle='dashed', linewidth=1, label=f'Average: {average_endpoint:.2f}')
axs[1].set_title('Distribution of Endpoint Values')
axs[1].set_xlabel('Stock Price')
axs[1].set_ylabel('Frequency')
axs[1].legend()

# Styling to match the given image
sns.set(style="whitegrid")
axs[0].axhline(y=average_endpoint, color='red', linestyle='dashed', linewidth=1, label='Average Endpoint')
axs[0].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Print the average of the endpoint values
print(f"The average endpoint value after {num_simulations} simulations is: {average_endpoint:.2f}")
