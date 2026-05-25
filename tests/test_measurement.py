import numpy as np
import open3d as o3d
import pytest
from src.measurement import compute_aabb, extract_dimensions


def _生成虚拟点云(n=200):
    pts = np.random.uniform(-0.02, 0.02, (n, 3)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_计算aabb返回包含所有键的字典():
    pcd = _生成虚拟点云()
    result = compute_aabb(pcd)
    for key in ('min_bound', 'max_bound', 'center', 'extent', 'length', 'width', 'height'):
        assert key in result, f"缺少键: {key}"


def test_计算aabb尺寸为非负():
    pcd = _生成虚拟点云(300)
    result = compute_aabb(pcd)
    assert result['length'] >= 0
    assert result['width'] >= 0
    assert result['height'] >= 0


def test_提取尺寸与aabb一致():
    pcd = _生成虚拟点云(300)
    result = extract_dimensions(pcd)
    aabb = compute_aabb(pcd)
    assert result['length'] == pytest.approx(aabb['length'], rel=1e-5)
    assert result['width'] == pytest.approx(aabb['width'], rel=1e-5)
    assert result['height'] == pytest.approx(aabb['height'], rel=1e-5)
