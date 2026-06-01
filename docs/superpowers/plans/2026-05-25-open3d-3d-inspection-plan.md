# Open3D 三维视觉检测系统 — 实现计划

> **面向执行代理：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务逐条实现。步骤使用 checkbox（`- [ ]`）语法跟踪进度。

**目标：** 构建离线三维视觉检测系统，对涡轮旋转片 PLY 点云进行预处理、配准对齐、尺寸测量和可视化。

**架构：** 模块化流水线 — 5 个源文件各司其职。数据以 `o3d.geometry.PointCloud` 对象在模块间传递。`main.py` 编排整个流程。

**技术栈：** Python 3.10+、Open3D 0.18+、NumPy、Matplotlib、pytest

**环境：** conda 环境 `open3d_pr`

---

### 任务 1：项目初始化与依赖安装

**涉及文件：**
- 新建：`requirements.txt`
- 新建：`tests/__init__.py`
- 新建：`src/__init__.py`

- [ ] **步骤 1：编写 requirements.txt**

```
open3d>=0.18.0
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.4.0
```

- [ ] **步骤 2：创建空的 __init__.py 文件**

```bash
touch tests/__init__.py src/__init__.py
```

- [ ] **步骤 3：安装依赖**

```bash
conda run -n open3d_pr pip install -r requirements.txt 2>&1
```
预期：全部安装成功（大部分已预装）。

- [ ] **步骤 4：验证 Open3D 可导入**

```bash
conda run -n open3d_pr python -c "import open3d; print(open3d.__version__)" 2>&1
```
预期：输出版本号，如 "0.18.0"

- [ ] **步骤 5：提交**

```bash
git add requirements.txt tests/__init__.py src/__init__.py
git commit -m "chore: 项目初始化，添加依赖配置"
```

---

### 任务 2：工具模块 — 日志、校验、异常

**涉及文件：**
- 新建：`src/utils.py`
- 新建：`tests/test_utils.py`

- [ ] **步骤 1：编写会失败的测试**

```python
import os
import tempfile
import pytest
from src.utils import validate_ply, setup_logging, FileFormatError, PointCloudValidationError


def test_校验合法ply文件通过():
    content = """ply
format ascii 1.0
comment test
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 1 1
2 2 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as f:
        f.write(content)
        path = f.name
    try:
        assert validate_ply(path) is True
    finally:
        os.unlink(path)


def test_校验不存在文件返回假():
    assert validate_ply("/nonexistent/file.ply") is False


def test_校验错误后缀返回假():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("not a ply")
        path = f.name
    try:
        assert validate_ply(path) is False
    finally:
        os.unlink(path)


def test_日志配置生成logger并写入():
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
        path = f.name
    try:
        logger = setup_logging(path)
        logger.info("test message")
        with open(path) as lf:
            content = lf.read()
        assert "test message" in content
    finally:
        os.unlink(path)


def test_文件格式异常可抛出():
    with pytest.raises(FileFormatError):
        raise FileFormatError("bad file")


def test_点云校验异常可抛出():
    with pytest.raises(PointCloudValidationError):
        raise PointCloudValidationError("empty point cloud")
```

- [ ] **步骤 2：运行测试，验证全部失败**

```bash
conda run -n open3d_pr python -m pytest tests/test_utils.py -v 2>&1
```
预期：全部 FAIL（模块未找到）

- [ ] **步骤 3：编写实现代码**

```python
import os
import logging


class FileFormatError(Exception):
    """文件格式错误异常"""
    pass


class PointCloudValidationError(Exception):
    """点云校验失败异常"""
    pass


def validate_ply(filepath: str) -> bool:
    """校验 PLY 文件路径是否有效"""
    if not os.path.isfile(filepath):
        return False
    if not filepath.lower().endswith('.ply'):
        return False
    return True


def setup_logging(log_path: str = "inspection.log") -> logging.Logger:
    """配置日志系统，同时输出到文件和控制台"""
    logger = logging.getLogger("open3d_inspection")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)
    return logger
```

- [ ] **步骤 4：运行测试，验证全部通过**

```bash
conda run -n open3d_pr python -m pytest tests/test_utils.py -v 2>&1
```
预期：6 条测试全部 PASS

- [ ] **步骤 5：提交**

```bash
git add src/utils.py tests/test_utils.py
git commit -m "feat: 添加工具模块 — 文件校验、日志、自定义异常"
```

---

### 任务 3：预处理模块 — 加载、降采样、滤波、法线估计

**涉及文件：**
- 新建：`src/preprocessing.py`
- 新建：`tests/test_preprocessing.py`

- [ ] **步骤 1：编写会失败的测试**

```python
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
```

- [ ] **步骤 2：运行测试，验证全部失败**

```bash
conda run -n open3d_pr python -m pytest tests/test_preprocessing.py -v 2>&1
```
预期：全部 FAIL

- [ ] **步骤 3：编写实现代码**

```python
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
```

- [ ] **步骤 4：运行测试，验证全部通过**

```bash
conda run -n open3d_pr python -m pytest tests/test_preprocessing.py -v 2>&1
```
预期：5 条测试全部 PASS

- [ ] **步驟 5：提交**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: 添加预处理模块 — PLY加载、降采样、滤波、法线估计"
```

---

### 任务 4：配准模块 — PCA 对齐 + ICP 精配准

**涉及文件：**
- 新建：`src/registration.py`
- 新建：`tests/test_registration.py`

- [ ] **步骤 1：编写会失败的测试**

```python
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
```

- [ ] **步骤 2：运行测试，验证全部失败**

```bash
conda run -n open3d_pr python -m pytest tests/test_registration.py -v 2>&1
```
预期：全部 FAIL

- [ ] **步骤 3：编写实现代码**

```python
import logging
import numpy as np
import open3d as o3d

logger = logging.getLogger("open3d_inspection")


def pca_align(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """PCA 主轴对齐——将点云旋转到主方向"""
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 按特征值降序排列
    order = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, order].T
    # 保证右手坐标系
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    aligned_pts = centered @ R.T
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(aligned_pts)
    logger.info(f"PCA 对齐完成。特征值: {eigenvalues[order]}")
    return result


def icp_fine_align(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float = 0.001,
    max_iteration: int = 2000
) -> o3d.geometry.PointCloud:
    """ICP 精配准——多视角拼接时使用"""
    reg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    logger.info(f"ICP: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.6f}")
    source.transform(reg.transformation)
    return source
```

- [ ] **步骤 4：运行测试，验证全部通过**

```bash
conda run -n open3d_pr python -m pytest tests/test_registration.py -v 2>&1
```
预期：3 条测试全部 PASS

- [ ] **步骤 5：提交**

```bash
git add src/registration.py tests/test_registration.py
git commit -m "feat: 添加配准模块 — PCA 对齐与 ICP 精配准"
```

---

### 任务 5：测量模块 — 包围盒、尺寸提取、截面、热力图

**涉及文件：**
- 新建：`src/measurement.py`
- 新建：`tests/test_measurement.py`

- [ ] **步骤 1：编写会失败的测试**

```python
import numpy as np
import open3d as o3d
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
```

- [ ] **步骤 2：运行测试，验证全部失败**

```bash
conda run -n open3d_pr python -m pytest tests/test_measurement.py -v 2>&1
```
预期：全部 FAIL

- [ ] **步骤 3：编写实现代码**

```python
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
        f"尺寸: L={info['length']:.4f} W={info['width']:.4f} H={info['height']:.4f} (米, AABB)"
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
```

- [ ] **步骤 4：运行测试，验证全部通过**

```bash
conda run -n open3d_pr python -m pytest tests/test_measurement.py -v 2>&1
```
预期：3 条测试全部 PASS

- [ ] **步骤 5：提交**

```bash
git add src/measurement.py tests/test_measurement.py
git commit -m "feat: 添加测量模块 — AABB、尺寸提取、截面、热力图"
```

---

### 任务 6：可视化模块 — 3D 渲染与标注

**涉及文件：**
- 新建：`src/visualization.py`
- 新建：`tests/test_visualization.py`

- [ ] **步骤 1：编写会失败的测试**

```python
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
```

- [ ] **步骤 2：运行测试，验证全部失败**

```bash
conda run -n open3d_pr python -m pytest tests/test_visualization.py -v 2>&1
```
预期：全部 FAIL

- [ ] **步骤 3：编写实现代码**

```python
import logging
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

logger = logging.getLogger("open3d_inspection")


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
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i, j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]], 'r-', linewidth=2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title)
    if dimensions:
        ax.text2D(0.02, 0.98,
                  f"长: {dimensions['length']*1000:.2f} mm\n"
                  f"宽: {dimensions['width']*1000:.2f} mm\n"
                  f"高: {dimensions['height']*1000:.2f} mm",
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
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('偏差 (度)')
    plt.tight_layout()
    logger.info(f"热力图可视化: {title}")
    return fig
```

- [ ] **步骤 4：运行测试，验证全部通过**

```bash
conda run -n open3d_pr python -m pytest tests/test_visualization.py -v 2>&1
```
预期：3 条测试全部 PASS

- [ ] **步骤 5：提交**

```bash
git add src/visualization.py tests/test_visualization.py
git commit -m "feat: 添加可视化模块 — 包围盒、尺寸标注、热力图"
```

---

### 任务 7：主入口 — 流水线编排

**涉及文件：**
- 新建：`main.py`

- [ ] **步骤 1：编写 main.py**

```python
"""Open3D 工业产品三维视觉检测系统"""
import argparse
import sys
from src.utils import setup_logging, FileFormatError, PointCloudValidationError
from src.preprocessing import load_ply, voxel_downsample, statistical_outlier_removal, estimate_normals
from src.registration import pca_align
from src.measurement import extract_dimensions, cross_section, deviation_heatmap
from src.visualization import draw_bounding_box, draw_heatmap


def main():
    parser = argparse.ArgumentParser(description='三维点云视觉检测系统')
    parser.add_argument('input', help='PLY 点云文件路径')
    parser.add_argument('--voxel-size', type=float, default=0.001, help='体素降采样尺寸 (米)')
    parser.add_argument('--output-dim', default='dimensions.txt', help='尺寸输出文件')
    parser.add_argument('--output-bbox', default='bbox.png', help='包围盒输出图像')
    parser.add_argument('--output-heatmap', default='heatmap.png', help='热力图输出图像')
    parser.add_argument('--log', default='inspection.log', help='日志文件路径')
    args = parser.parse_args()

    logger = setup_logging(args.log)
    logger.info("=== 三维检测开始 ===")
    logger.info(f"输入文件: {args.input}")

    try:
        # 1. 加载
        pcd = load_ply(args.input)
        logger.info(f"已加载 {len(pcd.points)} 个点")

        # 2. 预处理
        pcd = voxel_downsample(pcd, voxel_size=args.voxel_size)
        pcd = statistical_outlier_removal(pcd)
        pcd = estimate_normals(pcd)

        # 3. 对齐
        pcd = pca_align(pcd)

        # 4. 测量
        dims = extract_dimensions(pcd)

        # 保存尺寸
        with open(args.output_dim, 'w') as f:
            f.write(f"长度 (X): {dims['length']*1000:.3f} mm\n")
            f.write(f"宽度 (Y): {dims['width']*1000:.3f} mm\n")
            f.write(f"高度 (Z): {dims['height']*1000:.3f} mm\n")
        logger.info(f"尺寸已保存至 {args.output_dim}")

        # 截面示例（z=0 处）
        section = cross_section(pcd, axis='z', position=0.0)
        logger.info(f"z=0 处截面: {len(section)} 个点")

        # 偏差热力图
        deviations = deviation_heatmap(pcd)

        # 5. 可视化
        fig_bbox = draw_bounding_box(pcd, dims, title="涡轮旋转片 — 包围盒")
        fig_bbox.savefig(args.output_bbox, dpi=150)
        logger.info(f"包围盒图像已保存至 {args.output_bbox}")

        fig_heat = draw_heatmap(pcd, deviations, title="涡轮旋转片 — 偏差热力图")
        fig_heat.savefig(args.output_heatmap, dpi=150)
        logger.info(f"热力图图像已保存至 {args.output_heatmap}")

        print(f"\n=== 检测完成 ===")
        print(f"尺寸 (mm): 长={dims['length']*1000:.3f} 宽={dims['width']*1000:.3f} 高={dims['height']*1000:.3f}")
        print(f"输出文件: {args.output_dim}, {args.output_bbox}, {args.output_heatmap}")

    except FileFormatError as e:
        logger.error(str(e))
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except PointCloudValidationError as e:
        logger.error(str(e))
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"未知错误: {e}")
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **步骤 2：验证 main.py 可执行（无参数时显示帮助）**

```bash
conda run -n open3d_pr python main.py --help 2>&1
```
预期：输出 argparse 帮助信息，包含 `input`、`--voxel-size`、`--output-dim` 等参数说明。

- [ ] **步骤 3：用虚拟数据跑通完整流程**

```bash
conda run -n open3d_pr python -c "
import numpy as np
import open3d as o3d
pts = (np.random.randn(1000, 3) * 0.005).astype(np.float32)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.io.write_point_cloud('data/test_dummy.ply', pcd)
" 2>&1
```
然后：
```bash
conda run -n open3d_pr python main.py data/test_dummy.ply --voxel-size 0.001 2>&1
```
预期：输出尺寸信息，生成 `dimensions.txt`、`bbox.png`、`heatmap.png`。

- [ ] **步骤 4：提交**

```bash
git add main.py
git commit -m "feat: 添加主入口 — 完整检测流程编排"
```

---

### 任务 8：在真实涡轮片数据上运行

**涉及文件：**
- 无（使用已有数据）

- [ ] **步骤 1：定位真实 PLY 文件**

```bash
ls -la data/*.ply 2>&1
```

- [ ] **步骤 2：对真实数据运行检测流程**

```bash
conda run -n open3d_pr python main.py data/<真实文件名>.ply --voxel-size 0.001 2>&1
```
预期：输出尺寸（mm），生成检测结果文件。

- [ ] **步骤 3：验证输出结果**

```bash
ls -la dimensions.txt bbox.png heatmap.png inspection.log 2>&1
cat dimensions.txt
```
预期：所有文件均生成，尺寸数据合理。

- [ ] **步骤 4：提交结果**

```bash
git add dimensions.txt bbox.png heatmap.png
git commit -m "results: 涡轮旋转片检测结果"
```
