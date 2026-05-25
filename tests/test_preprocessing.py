import numpy as np
import open3d as o3d
import tempfile
import os
from src.preprocessing import load_ply, voxel_downsample, statistical_outlier_removal, estimate_normals


def _生成虚拟点云(n=200):
    """生成随机点云用于测试"""
    pts = np.random.randn(n, 3).astype(np.float32) * 0.01
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_加载ply返回点云对象():
    pcd = _生成虚拟点云(100)
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        path = f.name
    try:
        o3d.io.write_point_cloud(path, pcd)
        loaded = load_ply(path)
        assert isinstance(loaded, o3d.geometry.PointCloud)
        assert loaded.has_points()
    finally:
        os.unlink(path)


def test_体素降采样减少点数():
    pcd = _生成虚拟点云(500)
    down = voxel_downsample(pcd, voxel_size=0.005)
    assert len(down.points) <= len(pcd.points)
    assert len(down.points) > 0


def test_体素降采样保持点云结构():
    pcd = _生成虚拟点云(200)
    down = voxel_downsample(pcd, voxel_size=0.005)
    assert down.has_points()


def test_统计滤波去除离群点():
    pcd = _生成虚拟点云(200)
    # 添加明显离群点
    pts = np.asarray(pcd.points)
    pts[0] = [100, 100, 100]
    pcd.points = o3d.utility.Vector3dVector(pts)
    filtered = statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0)
    assert len(filtered.points) < len(pcd.points)


def test_法线估计添加法向量():
    pcd = _生成虚拟点云(100)
    result = estimate_normals(pcd)
    assert result.has_normals()
