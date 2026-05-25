import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免测试时弹出窗口
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import open3d as o3d

logger = logging.getLogger("open3d_inspection")

# 尝试使用系统中文字体
_CJK_FONT = None
for _font_name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi']:
    for _f in fm.fontManager.ttflist:
        if _font_name in _f.name:
            _CJK_FONT = _f.name
            break
    if _CJK_FONT:
        break

if _CJK_FONT:
    plt.rcParams['font.sans-serif'] = [_CJK_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    logger.info(f"Using CJK font: {_CJK_FONT}")
else:
    logger.warning("No CJK font found, Chinese characters may not render")


def draw_bounding_box(pcd: o3d.geometry.PointCloud, dimensions: dict = None, title: str = "包围盒"):
    """绘制点云及轴对齐包围盒"""
    aabb = pcd.get_axis_aligned_bounding_box()
    pts = np.asarray(pcd.points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 降采样显示以防止卡顿
    step = max(1, len(pts) // 5000)
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, alpha=0.5, c='steelblue')
    corners = np.asarray(aabb.get_box_points())
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]], 'r-', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    if dimensions:
        ax.text2D(0.02, 0.98,
                  f"长: {dimensions['length']:.2f} mm\n"
                  f"宽: {dimensions['width']:.2f} mm\n"
                  f"高: {dimensions['height']:.2f} mm",
                  transform=ax.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    logger.info(f"包围盒可视化: {title}")
    return fig


def draw_dimensions(pcd: o3d.geometry.PointCloud, dimensions: dict, title: str = "尺寸标注"):
    """绘制带尺寸标注的点云"""
    return draw_bounding_box(pcd, dimensions, title)


def draw_heatmap(pcd: o3d.geometry.PointCloud, deviations: np.ndarray, title: str = "偏差热力图"):
    """绘制法线偏差热力图"""
    pts = np.asarray(pcd.points)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    step = max(1, len(pts) // 5000)
    sc = ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                    c=deviations[::step], s=2, cmap='jet', alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('偏差 (度)')
    plt.tight_layout()
    logger.info(f"热力图可视化: {title}")
    return fig
