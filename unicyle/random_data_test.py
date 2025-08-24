import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import multivariate_normal

# Obstacle Parameters
num_obstacles = 5
points_per_cluster = 100
obstacle_spread = [0.01, 0.01]
obstacle_centers = np.array([[0,2],[1.5,2],[3,4],[1.7, 5],[3,6]])

# Generate points
points = []
for cx, cy in obstacle_centers:
    cluster = np.random.normal(loc=(cx, cy), scale=(obstacle_spread[0], obstacle_spread[1]), size=(points_per_cluster, 2))
    points.append(cluster)
points = np.vstack(points)

# Gaussian Mixture Modelling
bgm = BayesianGaussianMixture(n_components=num_obstacles).fit(points)
ob_means = bgm.means_
ob_covariances = bgm.covariances_
# print(ob_covariances[0])

# Destination
dest_center = [4,5]
dest_spread = [0.3, 0.3]
dest_points = np.random.normal(loc=(dest_center[0], dest_center[1]), scale=(dest_spread[0], dest_spread[1]), size=(points_per_cluster, 2))
dest_bgm = BayesianGaussianMixture(n_components=1).fit(dest_points)
dest_mean = dest_bgm.means_
dest_covariance = dest_bgm.covariances_


# Multivariate distributions
x,y = np.mgrid[0:7:0.1, 0:7:0.1]
pos = np.dstack((x,y))
three_d_points = multivariate_normal(ob_means[0], ob_covariances[0]).pdf(pos) +  multivariate_normal(ob_means[1], ob_covariances[1]).pdf(pos) + multivariate_normal(ob_means[2], ob_covariances[2]).pdf(pos) +  multivariate_normal(ob_means[3], ob_covariances[3]).pdf(pos) + multivariate_normal(ob_means[4], ob_covariances[4]).pdf(pos) -  multivariate_normal(dest_mean[0], dest_covariance[0]).pdf(pos)

# three_d_points = -multivariate_normal(dest_mean[0], dest_covariance[0]).pdf(pos)

# print(three_d_points.shape)
dx, dy = np.gradient(three_d_points)
dx = -dx
dy = -dy



fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, three_d_points, cmap='viridis')
skip = 1
plt.quiver(x[::skip], y[::skip], dx[::skip], dy[::skip], scale=30)
plt.contour(x, y, three_d_points, levels = 40)
plt.show()






# def f(x,y):
    
    
# Visualize
# plt.scatter(points[:,0], points[:,1], s=5, label="Obstacle points")
# plt.scatter(obstacle_centers[:,0], obstacle_centers[:,1], color='red', marker='x', label="Cluster centers")
# plt.legend()
# plt.title("Synthetic Obstacle Clusters")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.axis('equal')
# plt.show()

