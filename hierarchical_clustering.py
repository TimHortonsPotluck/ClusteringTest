import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

mean1 = [8, 3]
cov1 = [[3, 0], 
        [0, 3]]

mean2 = [0, 0]
cov2 = [[3, 0], 
        [0, 3]]

mean3 = [1, 1]
cov3 = [[.2, .05], 
        [.05, .05]]

mean4 = [5, 1]
cov4 = [[.1, -.05], 
        [-.05, .1]]

x1, y1 = rng.multivariate_normal(mean1, cov1, 100).T
x2, y2 = rng.multivariate_normal(mean2, cov2, 100).T
x3, y3 = np.exp(rng.multivariate_normal(mean3, cov3, 100).T)
x4, y4 = rng.multivariate_normal(mean4, cov4, 100).T

all_points = np.concatenate((np.column_stack((x1, y1)), 
                             np.column_stack((x2, y2)), 
                             np.column_stack((x3, y3)), 
                             np.column_stack((x4, y4))), axis=0)

def LN_dist(p1, p2, n):
    return (abs(p1[0] - p2[0])**n + abs(p1[1] - p2[1])**n)**(1/n)

def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def euclidean_dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# brute force hierarchical clustering

clusters = [all_points]






