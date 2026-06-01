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
        f"尺寸: L={info['length']:.4f} W={info['width']:.4f} H={info['height']:.4f} (mm, AABB)"
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


def detect_holes(
    pcd: o3d.geometry.PointCloud, axis: str = 'z', num_slices: int = 5, min_diameter: float = 0.5
) -> list:
    """检测圆孔并返回直径列表

    在多个截面上检测圆形孔洞，使用 RANSAC 圆拟合。
    """
    pts = np.asarray(pcd.points)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    other_axes = [i for i in range(3) if i != axis_idx]

    bounds = pts[:, axis_idx].min(), pts[:, axis_idx].max()
    offset = (bounds[1] - bounds[0]) * 0.1
    positions = np.linspace(bounds[0] + offset, bounds[1] - offset, max(num_slices, 2))

    all_diameters = []
    thickness = (bounds[1] - bounds[0]) / (num_slices * 3)

    for pos in positions:
        mask = np.abs(pts[:, axis_idx] - pos) < thickness
        slice_pts = pts[mask]
        if len(slice_pts) < 30:
            continue

        # 投影到 2D
        xy = slice_pts[:, other_axes]
        center_2d = xy.mean(axis=0)
        angles = np.arctan2(xy[:, 1] - center_2d[1], xy[:, 0] - center_2d[0])
        radii = np.linalg.norm(xy - center_2d, axis=1)

        # 按角度分桶，找半径最小值（孔边界）
        nbins = 72
        radius_profile = np.full(nbins, np.inf)
        bin_edges = np.linspace(-np.pi, np.pi, nbins + 1)
        for b in range(nbins):
            in_bin = (angles >= bin_edges[b]) & (angles < bin_edges[b + 1])
            if in_bin.any():
                radius_profile[b] = radii[in_bin].min()

        # 检测孔：角度桶中最小半径显著大于 0 的连续区域
        valid = radius_profile < np.inf
        if valid.sum() < nbins * 0.5:
            continue
        median_r = np.median(radius_profile[valid])

        # 找内边界（可能是孔的边缘）
        inner_edges = []
        gap_threshold = median_r * 0.3
        for b in range(nbins):
            r = radius_profile[b]
            if r < np.inf and abs(r - median_r) < gap_threshold:
                # 这个角度有内边界点
                angle = (bin_edges[b] + bin_edges[b + 1]) / 2
                inner_edges.append([center_2d[0] + r * np.cos(angle),
                                    center_2d[1] + r * np.sin(angle)])

        if len(inner_edges) < 12:
            continue

        # 用代数圆拟合内边界点
        inner = np.array(inner_edges)
        A = np.column_stack([inner[:, 0], inner[:, 1], np.ones(len(inner))])
        b = inner[:, 0]**2 + inner[:, 1]**2
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            cx, cy = sol[0] / 2, sol[1] / 2
            r_circle = np.sqrt(sol[2] + cx**2 + cy**2)
            diameter = 2 * r_circle
            if min_diameter < diameter < median_r * 3:
                all_diameters.append(float(diameter))
        except np.linalg.LinAlgError:
            continue

    if all_diameters:
        # 聚类去重
        all_diameters = np.array(all_diameters)
        unique = []
        for d in sorted(all_diameters):
            if not unique or abs(d - unique[-1]) > 0.5:
                unique.append(d)
        logger.info(f"检测到 {len(unique)} 个孔洞, 直径: {[f'{d:.2f}' for d in unique]} mm")
        return unique
    return []


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
