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


def fpfh_ransac_align(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 1.0,
    distance_threshold: float = 1.5,
) -> o3d.geometry.PointCloud:
    """FPFH+RANSAC 粗配准 —— 基于特征匹配的全局对齐"""
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

    # 计算 FPFH 特征（增大搜索半径以获取更多匹配点）
    feature_radius = voxel_size * 10
    fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100)
    )
    fpfh_tgt = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100)
    )

    # RANSAC 全局配准
    ransac_dist = distance_threshold * voxel_size
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, fpfh_src, fpfh_tgt, True,
        ransac_dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
    )
    source.transform(result.transformation)
    logger.info(f"FPFH+RANSAC: fitness={result.fitness:.4f}, "
                f"rmse={result.inlier_rmse:.4f}, "
                f"correspondence={len(result.correspondence_set)}")
    return source


def icp_fine_align(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float = 1.0,
    max_iteration: int = 2000
) -> o3d.geometry.PointCloud:
    """ICP 精配准 —— 精细化对齐"""
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iteration
        )
    )
    logger.info(f"ICP: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.6f}")
    source.transform(reg.transformation)
    return source
