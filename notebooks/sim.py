import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Parameters
dim = 300        # Dimensionality of the space
timesteps = 200  # Number of time steps
runs = 1000      # Simulation repetitions
angle_deg = 66   # Angle between vectors (cosine similarity ≈ 0.4)

# Create two unit vectors with a specific cosine similarity
def generate_nonorthogonal_vectors(dim, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    v1 = np.random.randn(dim)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(dim)
    v2 -= v2 @ v1 * v1  # make orthogonal to v1
    v2 /= np.linalg.norm(v2)
    v2 = np.cos(angle_rad) * v1 + np.sin(angle_rad) * v2
    return v1, v2

correlations = []

for _ in range(runs):
    v1, v2 = generate_nonorthogonal_vectors(dim, angle_deg)

    # Generate two independent latent processes
    x = np.random.randn(timesteps)
    y = np.random.randn(timesteps)

    # Create matrix of latent processes embedded in high dimensions
    # Each row is a timestep, columns are the dim space
    signals = np.outer(x, v1) + np.outer(y, v2)

    # Project back onto v1 and v2
    proj1 = signals @ v1
    proj2 = signals @ v2

    # Compute correlation between the projections
    r, _ = pearsonr(proj1, proj2)
    correlations.append(r)

# Plot histogram
plt.hist(correlations, bins=50, edgecolor='k')
plt.title(f"Distribution of Correlations (angle = {angle_deg}°)")
plt.xlabel("Pearson r")
plt.ylabel("Frequency")
plt.axvline(np.mean(correlations), color='red', label=f"Mean r = {np.mean(correlations):.2f}")
plt.legend()
plt.show()
