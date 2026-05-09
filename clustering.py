import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

mean1 = [8, 3]
cov1 = [[1, 0], 
        [0, 1]]

mean2 = [3, 5]
cov2 = [[1, 0], 
        [0, 1]]

mean3 = [1, 1]
cov3 = [[.2, .05], 
        [.05, .05]]

mean4 = [5, 1]
cov4 = [[.1, -.05], 
        [-.05, .1]]

x1, y1 = rng.multivariate_normal(mean1, cov1, 1000).T
x2, y2 = rng.multivariate_normal(mean2, cov2, 1000).T
x3, y3 = np.exp(rng.multivariate_normal(mean3, cov3, 1000).T)
x4, y4 = rng.multivariate_normal(mean4, cov4, 100).T

all_points = np.concatenate((np.column_stack((x1, y1)), 
                             np.column_stack((x2, y2)), 
                             np.column_stack((x3, y3)),
                             np.column_stack((x4, y4))), axis=0)

def LN_dist(p1, p2, n):
    return (abs(p1[0] - p2[0])**n + abs(p1[1] - p2[1])**n)**(1/n)

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

num_clusters = 2
center_points = all_points[rng.choice(len(all_points), num_clusters, replace=False)]

num_iterations = 50
for i in range(num_iterations):
    clusters = [np.array([]).reshape(0, 2) for i in range(num_clusters)]
    next_centers = np.zeros_like(center_points)
    for point in all_points:
        distances = [LN_dist(point, center, 2) for center in center_points]
        closest_center_index = np.argmin(distances)
        clusters[closest_center_index] = np.vstack((clusters[closest_center_index], point))

    for j in range(num_clusters):
        next_centers[j] = np.mean(clusters[j], axis=0)
    if np.allclose(center_points, next_centers):
        print(f"Converged after {i} iterations")
        break
    center_points = next_centers
# print(clusters)
for cluster in clusters:
    plt.scatter(cluster[:, 0], cluster[:, 1], s=1)
plt.scatter(center_points[:, 0], center_points[:, 1], c='black', marker='o', s=100)
plt.axis('equal')
plt.show()










