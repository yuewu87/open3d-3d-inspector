import open3d as o3d
from src.utils import validate_ply, FileFormatError


def load_ply(filepath: str) -> o3d.geometry.PointCloud:
    """加载 PLY 点云文件"""
    if not validate_ply(filepath):
        raise FileFormatError(f"无效的 PLY 文件: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd is None or not pcd.has_points():
        raise FileFormatError(f"无法加载点云: {filepath}")
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.001) -> o3d.geometry.PointCloud:
    """体素降采样"""
    down = pcd.voxel_down_sample(voxel_size)
    return down


def statistical_outlier_removal(pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """统计离群点滤波"""
    filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.01, max_nn: int = 30) -> o3d.geometry.PointCloud:
    """法线估计"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd
