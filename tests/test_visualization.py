import numpy as np
import open3d as o3d
from src.visualization import draw_bounding_box, draw_dimensions, draw_heatmap


def _生成虚拟点云(n=200):
    pts = np.random.uniform(-0.01, 0.01, (n, 3)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_绘制包围盒返回图形对象():
    pcd = _生成虚拟点云()
    fig = draw_bounding_box(pcd, title="test")
    assert fig is not None


def test_绘制尺寸标注返回图形对象():
    result = {
        'length': 0.02, 'width': 0.015, 'height': 0.01,
        'center': np.array([0, 0, 0])
    }
    pcd = _生成虚拟点云()
    fig = draw_dimensions(pcd, result, title="test")
    assert fig is not None


def test_绘制热力图返回图形对象():
    pcd = _生成虚拟点云()
    deviations = np.random.uniform(0, 45, len(pcd.points))
    fig = draw_heatmap(pcd, deviations, title="test")
    assert fig is not None
