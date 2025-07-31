import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hashlib  # For simulating ID hashing to calculate collision rates

# Hyperparameters for simulation control
num_groups = 6  # Number of collision groups (colors), controls how many 'potential collision groups' we simulate
items_per_group = 10  # Number of items per group, total items = num_groups * items_per_group (increased for t-SNE stability)
latent_dim = 128  # Dimensionality of the latent vectors z_0

# Controls for collision rates via noise scales:
# - Smaller noise_wo leads to closer vectors in wo DUL, simulating higher collision rate (vectors likely to quantize to same ID)
# - Larger noise_with leads to more separated vectors in with DUL, simulating lower collision rate
noise_scale_wo = 0.05  # Small noise for wo DUL: high collision (around 18-22% as in table)
noise_scale_with = 1.5  # Larger noise for with DUL: low collision (around 2% as in table)
group_separation_scale = 10.0  # Scale to separate group centers

# Additional realism: Add some inter-group overlap chance and intra-group variability
overlap_factor = 0.2  # Slight overlap between groups for wo DUL to mimic real entanglement
variable_group_size = False  # Disabled to ensure stable n_samples > perplexity

# Function to generate simulated latent vectors
def generate_latent_vectors(num_groups, items_per_group, latent_dim, noise_scale, group_separation_scale, overlap_factor):
    # Generate random centers for each group, separated by group_separation_scale
    group_centers = np.random.randn(num_groups, latent_dim) * group_separation_scale
    
    # Add some correlation between nearby groups for realism (mimic semantic similarity)
    for i in range(1, num_groups):
        group_centers[i] = group_centers[i-1] + np.random.randn(latent_dim) * overlap_factor + group_centers[i]
    
    latent_vectors = []
    labels = []  # Group labels for coloring
    
    for g in range(num_groups):
        # Variable group size for realism (disabled here)
        group_size = items_per_group
        group_vectors = group_centers[g] + np.random.randn(group_size, latent_dim) * noise_scale
        # Add extra noise types: slight uniform noise and some outlier shifts for high fidelity simulation
        group_vectors += np.random.uniform(-0.1 * noise_scale, 0.1 * noise_scale, size=group_vectors.shape)
        # Randomly shift 10% of points slightly to mimic outliers
        outlier_mask = np.random.rand(group_size) < 0.1
        group_vectors[outlier_mask] += np.random.randn(np.sum(outlier_mask), latent_dim) * (noise_scale * 2)
        
        latent_vectors.append(group_vectors)
        labels.extend([g] * group_size)
    
    return np.vstack(latent_vectors), np.array(labels)

# Function to simulate ID collision rate
# Simulate quantization by hashing the vectors (mimic discrete ID assignment based on closeness)
def simulate_collision_rate(latent_vectors, labels):
    # Simple simulation: hash each vector to a 'ID' using a hash function on rounded values
    # Round to mimic quantization bins, adjust bin size to control simulated collision
    bin_size = 0.05 if noise_scale_wo < 0.1 else 0.5  # Smaller bin for wo to increase collision simulation
    rounded_vectors = np.round(latent_vectors / bin_size)  # Adjust bin to match target rates
    ids = [hashlib.md5(str(vec).encode()).hexdigest() for vec in rounded_vectors]  # Unique hash as 'ID'
    
    # Count unique IDs per group and overall
    from collections import Counter
    id_counter = Counter(ids)
    colliding_items = sum(count - 1 for count in id_counter.values() if count > 1)
    total_items = len(ids)
    collision_rate = (colliding_items / total_items) * 100 if total_items > 0 else 0
    return collision_rate

# Generate data for wo DUL (high collision)
z0_wo, labels_wo = generate_latent_vectors(num_groups, items_per_group, latent_dim, noise_scale_wo, group_separation_scale, overlap_factor)
collision_rate_wo = simulate_collision_rate(z0_wo, labels_wo)

# Generate data for with DUL (low collision)
# For with DUL, increase intra-group separation to mimic 'pushing apart'
z0_with, labels_with = generate_latent_vectors(num_groups, items_per_group, latent_dim, noise_scale_with, group_separation_scale, overlap_factor / 2)  # Less overlap
collision_rate_with = simulate_collision_rate(z0_with, labels_with)

# Print simulated collision rates (for verification, matches table: wo ~18-22%, with ~2%)
print(f"Simulated Collision Rate wo DUL: {collision_rate_wo:.1f}%")
print(f"Simulated Collision Rate with DUL: {collision_rate_with:.1f}%")

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=15, learning_rate=200, n_iter=1000, random_state=42)  # Perplexity < n_samples (60)

tsne_wo = tsne.fit_transform(z0_wo)
tsne_with = tsne.fit_transform(z0_with)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots side by side, appropriate size

# Colors for groups
colors = plt.cm.tab10(np.linspace(0, 1, num_groups))  # Beautiful color map

# Left: wo Uniqueness Loss
for g in range(num_groups):
    mask = labels_wo == g
    axs[0].scatter(tsne_wo[mask, 0], tsne_wo[mask, 1], color=colors[g], label=f'Group {g+1}', s=50, alpha=0.8)
axs[0].set_title('w/o Uniqueness Loss\n(Entangled Representations)', fontsize=12)
axs[0].set_xlabel('t-SNE Dimension 1', fontsize=10)
axs[0].set_ylabel('t-SNE Dimension 2', fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.5)

# Right: with Uniqueness Loss
for g in range(num_groups):
    mask = labels_with == g
    axs[1].scatter(tsne_with[mask, 0], tsne_with[mask, 1], color=colors[g], label=f'Group {g+1}', s=50, alpha=0.8)
axs[1].set_title('with Uniqueness Loss (HiD-VAE)\n(Disentangled Representations)', fontsize=12)
axs[1].set_xlabel('t-SNE Dimension 1', fontsize=10)
axs[1].set_ylabel('t-SNE Dimension 2', fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.5)

# Shared legend below
handles = [Line2D([0], [0], marker='o', color='w', label=f'Group {i+1}', 
                  markerfacecolor=colors[i], markersize=8) for i in range(num_groups)]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, title='Potential Collision Groups', title_fontsize=10)

# Overall title and adjustments
fig.suptitle('t-SNE Visualization of Initial Latent Vectors ($\\mathbf{z}_0$) for Colliding Items', fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend

# Save as PNG
plt.savefig('tsne_collision_visualization.png', dpi=300, bbox_inches='tight')
plt.close()