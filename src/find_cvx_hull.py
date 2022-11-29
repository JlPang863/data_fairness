from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt


def func_cvx_bound(points):
    hull = ConvexHull(points)
    vertices_all = hull.vertices[::-1]

    left = np.argmin(points[:,0])
    right = np.argmax(points[:,0])
    loc_left = np.where(vertices_all==left)[0][0]
    vertices_all = np.concatenate((vertices_all[loc_left:], vertices_all[:loc_left]))
    # points_new = np.concatenate((points[loc_left:], points[:loc_left]))

    loc_right = np.where(vertices_all==right)[0][0]
    vertices_bound = vertices_all[:loc_right+1]
    return vertices_bound

if __name__ == "__main__":  # unit test
    rng = np.random.default_rng()
    points = rng.random((30, 2))   # 30 random points in 2-D

    vertices_bound = func_cvx_bound(points)
    plt.plot(points[:,0], points[:,1], 'o')
    plt.plot(points[vertices_bound,0], points[vertices_bound,1], 'r--', lw=2)
    plt.plot(points[vertices_bound[0],0], points[vertices_bound[0],1], 'ro')
    plt.show()