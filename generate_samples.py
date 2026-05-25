"""生成不同几何体的测试 PLY 文件，用于验证三维检测系统效果。

用法：python generate_samples.py
输出：data/sample_box.ply, sample_cylinder.ply, sample_sphere.ply, sample_cone.ply
"""
import numpy as np
import open3d as o3d


def add_noise(pts: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """添加高斯噪声"""
    noise = np.random.randn(*pts.shape).astype(np.float32) * sigma
    return pts + noise


def save_pcd(pts: np.ndarray, path: str, color: list = None):
    """保存点云为 PLY 文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    if color:
        colors = np.tile(np.array(color, dtype=np.float32) / 255.0, (len(pts), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    # 估算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    o3d.io.write_point_cloud(path, pcd)
    print(f"  已保存: {path} ({len(pts)} 点)")


def generate_box():
    """生成长方体点云 (50x30x20 mm)"""
    print("1. 生成长方体 (box)...")
    n = 5000
    pts = np.random.uniform(0, [50, 30, 20], (n, 3)).astype(np.float32)
    save_pcd(pts, "data/sample_box.ply", color=[100, 150, 200])


def generate_cylinder():
    """生成圆柱体点云 (直径30mm, 高60mm)"""
    print("2. 生成圆柱体 (cylinder)...")
    n = 5000
    r = 15.0
    h = 60.0
    # 随机半径分布（内部填充）
    radii = r * np.sqrt(np.random.uniform(0, 1, n))
    angles = np.random.uniform(0, 2 * np.pi, n)
    heights = np.random.uniform(0, h, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = heights
    pts = np.column_stack([x, y, z])
    save_pcd(pts, "data/sample_cylinder.ply", color=[200, 100, 100])


def generate_sphere():
    """生成球体点云 (直径40mm)"""
    print("3. 生成球体 (sphere)...")
    n = 5000
    r = 20.0
    # 球体内均匀填充
    radii = r * np.cbrt(np.random.uniform(0, 1, n))
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = radii * np.sin(phi) * np.cos(theta)
    y = radii * np.sin(phi) * np.sin(theta)
    z = radii * np.cos(phi)
    pts = np.column_stack([x, y, z])
    save_pcd(pts, "data/sample_sphere.ply", color=[100, 200, 100])


def generate_cone():
    """生成圆锥体点云 (底面直径40mm, 高50mm)"""
    print("4. 生成圆锥体 (cone)...")
    n = 5000
    r_base = 20.0
    h = 50.0
    # 沿高度均匀分布
    heights = np.random.uniform(0, h, n)
    # 半径随高度线性减小
    max_r = r_base * (1 - heights / h)
    radii = max_r * np.sqrt(np.random.uniform(0, 1, n))
    angles = np.random.uniform(0, 2 * np.pi, n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = heights
    pts = np.column_stack([x, y, z])
    save_pcd(pts, "data/sample_cone.ply", color=[200, 200, 100])


if __name__ == '__main__':
    import os
    os.makedirs("data", exist_ok=True)
    generate_box()
    generate_cylinder()
    generate_sphere()
    generate_cone()
    print("\n全部样本生成完成！")
    print("运行示例：python main.py data/sample_box.ply")
