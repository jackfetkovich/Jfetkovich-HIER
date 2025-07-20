import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

# Parameters
num_clusters = 5
points_per_cluster = 100
spreadX = 0.25  # Standard deviation of each cluster
spreadY = 0.7

# Random cluster centers in a 10x10 area
centers = np.array([[1,2],[1.5,2],[3,4],[1.7, 5],[3,6]])

# Generate points
points = []
for cx, cy in centers:
    cluster = np.random.normal(loc=(cx, cy), scale=(spreadX, spreadY), size=(points_per_cluster, 2))
    points.append(cluster)

points = np.vstack(points)

# Gaussian Mixture Modelling
bgm = BayesianGaussianMixture(n_components=num_clusters).fit(points)
print(bgm.means_)

# Visualize
plt.scatter(points[:,0], points[:,1], s=5, label="Obstacle points")
plt.scatter(centers[:,0], centers[:,1], color='red', marker='x', label="Cluster centers")
plt.legend()
plt.title("Synthetic Obstacle Clusters")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.show()

