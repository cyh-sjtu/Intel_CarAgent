import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation

# 加载配置
from caragent_agent.config.config import config

def depth_to_pointcloud(depth_img, intrinsic):
    # 生成点云从深度图
    if len(depth_img.shape) > 2:
        depth_img = depth_img[:, :, 0]
    depth = depth_img.astype(np.float32)

    h, w = depth.shape
    j, i = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # 从config读取深度阈值
    depth_min = float(config.get('pointcloud_valid_depth_min', 0))
    depth_max = float(config.get('pointcloud_valid_depth_max', 1000))
    z_full = depth
    valid_full = (z_full > depth_min) & (z_full < depth_max)
    z_full = z_full[valid_full]
    j_full = j[valid_full]
    i_full = i[valid_full]

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    x_full = (i_full - cx) * z_full / fx
    y_full = (j_full - cy) * z_full / fy

    points_full = np.stack((x_full, y_full, z_full), axis=-1)
    return points_full

def transform_points(points, tf, z_min=None, z_max=None):
    """
    将点云从相机坐标系变换到世界坐标系
    :param points: N×3 numpy数组
    :param tf: 4×4 变换矩阵
    :return: N×3 numpy数组
    """
    N = points.shape[0]
    homo_pts = np.hstack([points, np.ones((N,1))])  # 齐次坐标
    pts_world = (tf @ homo_pts.T).T
    if z_min is not None:
        mask = pts_world[:, 2] >= z_min
        pts_world = pts_world[mask]

    if z_max is not None:
        mask = pts_world[:, 2] <= z_max
        pts_world = pts_world[mask]
    return pts_world[:, :3]

def compute_min_vertical_bbox(points: np.ndarray) -> np.ndarray:
    """
    计算点集的最小竖直长方体（底面为 XY 平面上的最小矩形，由 OpenCV 的 minAreaRect 给出），
    返回 8 个顶点 (8,3)。当 points 为空时返回 zeros。
    """
    if points is None or points.shape[0] == 0:
        return np.zeros((8, 3), dtype=np.float32)

    pts_xy = points[:, :2].astype(np.float32)

    # cv2.minAreaRect 对于少量点仍能工作（w 或 h 可能为 0）
    rect = cv2.minAreaRect(pts_xy)  # ((cx, cy), (w, h), angle)
    box2d = cv2.boxPoints(rect)     # (4,2) 顺序构成矩形
    box2d = np.array(box2d, dtype=np.float32)

    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))

    lower = np.hstack([box2d, np.full((4, 1), z_min, dtype=np.float32)])
    upper = np.hstack([box2d, np.full((4, 1), z_max, dtype=np.float32)])
    bbox_vertices = np.vstack([lower, upper])

    return bbox_vertices

def remove_noise(points, nb_neighbors=None, std_ratio=None):
    """
    Removes noise from a point cloud using statistical outlier removal.
    :param points: N×3 numpy array.
    :param nb_neighbors: Number of neighbors to use for mean distance estimation.
    :param std_ratio: Standard deviation ratio.
    :return: Denoised points as a numpy array.
    """
    if points.shape[0] == 0:
        return points
    if nb_neighbors is None:
        nb_neighbors = int(config.get('pointcloud_denoise_nb_neighbors', 30))
    if std_ratio is None:
        std_ratio = float(config.get('pointcloud_denoise_std_ratio', 1.0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

def generate_pointcloud_from_depth(depth_path, intrinsic, orientation, position):
    # Load depth image
    depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        return np.array([])
    depth_image = depth_image.astype(np.float32) / 1000.0

    # Generate point cloud from depth image
    rotation_matrix = Rotation.from_quat(orientation).as_matrix()
    tf_cam_to_world = np.eye(4)
    tf_cam_to_world[:3, :3] = rotation_matrix
    tf_cam_to_world[:3, 3] = position

    cam_points_full = depth_to_pointcloud(depth_image, intrinsic)

    # Point cloud to NED coordinate system (same as pose coordinate system)
    cam_points_full_ned = np.zeros_like(cam_points_full)
    cam_points_full_ned[:, 0] = cam_points_full[:, 2]      # z -> x
    cam_points_full_ned[:, 1] = cam_points_full[:, 0]     # x -> y
    cam_points_full_ned[:, 2] = cam_points_full[:, 1]     # y -> z

    world_points_full = transform_points(cam_points_full_ned, tf_cam_to_world)
    # Remove noise points using config参数
    nb_neighbors = int(config.get('pointcloud_denoise_nb_neighbors', 30))
    std_ratio = float(config.get('pointcloud_denoise_std_ratio', 1.0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points_full)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # Remove points near the camera (e.g., within config距离)
    remove_near_dist = float(config.get('pointcloud_remove_near_distance', 0.5))
    distances = np.linalg.norm(np.asarray(pcd.points) - position, axis=1)
    mask = distances > remove_near_dist
    pcd = pcd.select_by_index(np.where(mask)[0])
    world_points_full = np.asarray(pcd.points)
    return world_points_full