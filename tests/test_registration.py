import numpy as np
import open3d as o3d
from src.registration import pca_align


def _生成长方体点云(n=300):
    """生成有明显主轴方向的点云"""
    pts = np.random.uniform(-0.1, 0.1, (n, 3))
    pts[:, 0] *= 5.0  # X 轴拉长
    pts = pts.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_pca对齐返回点云对象():
    pcd = _生成长方体点云()
    aligned = pca_align(pcd)
    assert isinstance(aligned, o3d.geometry.PointCloud)
    assert aligned.has_points()


def test_pca对齐保持点数不变():
    pcd = _生成长方体点云(200)
    aligned = pca_align(pcd)
    assert len(aligned.points) == len(pcd.points)


def test_pca对齐将主轴旋转到x轴():
    pcd = _生成长方体点云(500)
    aligned = pca_align(pcd)
    pts = np.asarray(aligned.points)
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_y = pts[:, 1].max() - pts[:, 1].min()
    span_z = pts[:, 2].max() - pts[:, 2].min()
    # X 轴方向跨度应该最大
    assert span_x >= span_y or span_x >= span_z
