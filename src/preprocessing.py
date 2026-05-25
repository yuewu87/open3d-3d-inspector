import logging
import open3d as o3d
from src.utils import validate_ply, FileFormatError

logger = logging.getLogger("open3d_inspection")


def load_ply(filepath: str) -> o3d.geometry.PointCloud:
    """加载 PLY 点云文件"""
    if not validate_ply(filepath):
        raise FileFormatError(f"无效的 PLY 文件: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd is None or not pcd.has_points():
        raise FileFormatError(f"无法加载点云: {filepath}")
    logger.info(f"已加载 PLY: {filepath}, 点数: {len(pcd.points)}")
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.001) -> o3d.geometry.PointCloud:
    """体素降采样"""
    before = len(pcd.points)
    down = pcd.voxel_down_sample(voxel_size)
    logger.info(f"体素降采样: {before} -> {len(down.points)} (voxel={voxel_size})")
    return down


def statistical_outlier_removal(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """统计离群点滤波"""
    before = len(pcd.points)
    filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    removed = before - len(filtered.points)
    logger.info(f"离群点滤波: {before} -> {len(filtered.points)} (移除 {removed} 点)")
    return filtered


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.01, max_nn: int = 30) -> o3d.geometry.PointCloud:
    """法线估计"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    logger.info(f"法线估计完成 (radius={radius}, max_nn={max_nn})")
    return pcd
