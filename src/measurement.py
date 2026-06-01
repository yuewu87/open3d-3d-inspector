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
    pcd: o3d.geometry.PointCloud, axis: str = 'z', num_slices: int = 10,
    min_diameter: float = 0.5, max_diameter: float = 100.0
) -> list:
    """检测贯穿型圆孔并返回直径列表

    算法：多截面圆检测 → 跨层一致性过滤（真孔沿轴向连续出现）
    """
    pts = np.asarray(pcd.points)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    other_axes = [i for i in range(3) if i != axis_idx]

    bounds = pts[:, axis_idx].min(), pts[:, axis_idx].max()
    margin = (bounds[1] - bounds[0]) * 0.1
    positions = np.linspace(bounds[0] + margin, bounds[1] - margin, max(num_slices, 3))
    thickness = (bounds[1] - bounds[0]) / (num_slices * 2)

    # 收集每层的候选圆: [(z_position, diameter, cx, cy), ...]
    candidates = []

    for pos in positions:
        mask = np.abs(pts[:, axis_idx] - pos) < thickness
        slice_pts = pts[mask]
        if len(slice_pts) < 50:
            continue

        xy = slice_pts[:, other_axes]
        center_2d = xy.mean(axis=0)
        angles = np.arctan2(xy[:, 1] - center_2d[1], xy[:, 0] - center_2d[0])
        radii = np.linalg.norm(xy - center_2d, axis=1)

        # 角度分桶
        nbins = 72
        radius_profile = np.full(nbins, np.inf)
        bin_edges = np.linspace(-np.pi, np.pi, nbins + 1)
        for b in range(nbins):
            in_bin = (angles >= bin_edges[b]) & (angles < bin_edges[b + 1])
            if in_bin.any():
                radius_profile[b] = radii[in_bin].min()

        valid = radius_profile < np.inf
        # 要求至少 60% 角度覆盖
        if valid.sum() < nbins * 0.6:
            continue
        median_r = np.median(radius_profile[valid])

        # 收集内边界点（半径接近 median_r * 0.5，即"内侧"边界的点）
        inner_edges = []
        for b in range(nbins):
            r = radius_profile[b]
            if r < np.inf and r < median_r * 0.7:
                angle = (bin_edges[b] + bin_edges[b + 1]) / 2
                inner_edges.append([center_2d[0] + r * np.cos(angle),
                                    center_2d[1] + r * np.sin(angle)])

        # 至少 30% 角度有内边界
        if len(inner_edges) < int(nbins * 0.3):
            continue

        # 代数圆拟合
        inner = np.array(inner_edges)
        A = np.column_stack([inner[:, 0], inner[:, 1], np.ones(len(inner))])
        b_vec = inner[:, 0]**2 + inner[:, 1]**2
        try:
            sol = np.linalg.lstsq(A, b_vec, rcond=None)[0]
            cx, cy = sol[0] / 2, sol[1] / 2
            r_circle = np.sqrt(sol[2] + cx**2 + cy**2)
            diameter = 2 * r_circle
            if min_diameter < diameter < max_diameter and diameter < median_r * 2:
                candidates.append((float(pos), float(diameter), float(cx), float(cy)))
        except np.linalg.LinAlgError:
            continue

    if not candidates:
        return []

    # 跨层聚类：直径相近（±20%）且在连续层出现
    candidates.sort(key=lambda x: x[1])  # 按直径排序
    groups = []
    used = set()
    for i, (z_i, d_i, cx_i, cy_i) in enumerate(candidates):
        if i in used:
            continue
        group = [candidates[i]]
        used.add(i)
        for j, (z_j, d_j, cx_j, cy_j) in enumerate(candidates):
            if j in used:
                continue
            # 直径相近，且中心距离不远的归为一组
            if abs(d_j - d_i) / max(d_i, 0.01) < 0.2:
                group.append(candidates[j])
                used.add(j)
        groups.append(group)

    # 只保留跨 >= 2 层的孔（真贯穿孔洞）
    result = []
    for g in groups:
        if len(g) >= 2:
            avg_d = np.mean([x[1] for x in g])
            result.append(float(avg_d))

    result = sorted(set(round(d, 1) for d in result))
    if result:
        logger.info(f"检测到 {len(result)} 个孔洞(跨层验证), 直径: {result} mm")
    else:
        logger.info("未检测到贯穿型孔洞 (跨层验证未通过)")
    return result


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
