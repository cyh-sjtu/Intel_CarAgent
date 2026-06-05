import numpy as np

def perpendicular_distance_3d(point, start, end):
    """计算点到直线的三维垂直距离"""
    p = np.array(point)
    a = np.array(start)
    b = np.array(end)
    if np.allclose(a, b):
        return np.linalg.norm(p - a)
    return np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(a - b)

def simplify_path_dp_3d(points, tol):
    """Douglas-Peucker 三维简化"""
    if len(points) <= 2:
        return points
    max_dist = 0
    idx = 0
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance_3d(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            idx = i
    if max_dist > tol:
        left = simplify_path_dp_3d(points[:idx + 1], tol)
        right = simplify_path_dp_3d(points[idx:], tol)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]