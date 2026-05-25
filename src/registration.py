import logging
import numpy as np
import open3d as o3d

logger = logging.getLogger("open3d_inspection")


def pca_align(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """PCA 主轴对齐——将点云旋转到主方向"""
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 按特征值降序排列
    order = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, order].T
    # 保证右手坐标系
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    aligned_pts = centered @ R.T
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(aligned_pts)
    logger.info(f"PCA 对齐完成。特征值: {eigenvalues[order]}")
    return result


def icp_fine_align(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float = 0.001,
    max_iteration: int = 2000
) -> o3d.geometry.PointCloud:
    """ICP 精配准——多视角拼接时使用"""
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    logger.info(f"ICP: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.6f}")
    source.transform(reg.transformation)
    return source
