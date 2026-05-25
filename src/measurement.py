import logging
import numpy as np
import open3d as o3d

logger = logging.getLogger("open3d_inspection")


def compute_aabb(pcd: o3d.geometry.PointCloud) -> dict:
    """计算轴对齐包围盒，返回长宽高等信息"""
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    center = aabb.get_center()
    minb = aabb.get_min_bound()
    maxb = aabb.get_max_bound()
    return {
        'min_bound': np.asarray(minb),
        'max_bound': np.asarray(maxb),
        'center': np.asarray(center),
        'extent': np.asarray(extent),
        'length': float(extent[0]),
        'width': float(extent[1]),
        'height': float(extent[2]),
    }


def extract_dimensions(pcd: o3d.geometry.PointCloud) -> dict:
    """提取关键尺寸（AABB 长宽高）"""
    info = compute_aabb(pcd)
    logger.info(
        f"尺寸: L={info['length']:.4f} W={info['width']:.4f} H={info['height']:.4f} (米, AABB)"
    )
    return {
        'length': info['length'],
        'width': info['width'],
        'height': info['height'],
        'center': info['center'],
    }


def cross_section(
    pcd: o3d.geometry.PointCloud, axis: str = 'z', position: float = 0.0, thickness: float = 0.001
) -> np.ndarray:
    """提取点云截面——在指定轴位置切厚度为 thickness 的薄片"""
    pts = np.asarray(pcd.points)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    mask = np.abs(pts[:, axis_idx] - position) < thickness / 2.0
    logger.info(f"截面 axis={axis} pos={position}: {mask.sum()} 个点在切片内")
    return pts[mask]


def deviation_heatmap(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """基于法线一致性生成偏差热力图数据（偏离角度，单位：度）"""
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    normals = np.asarray(pcd.normals)
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    directions = pts - center
    directions /= (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10)
    dot = np.abs(np.sum(normals * directions, axis=1))
    deviations = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    logger.info(f"热力图: 平均偏差={deviations.mean():.2f} 度")
    return deviations
