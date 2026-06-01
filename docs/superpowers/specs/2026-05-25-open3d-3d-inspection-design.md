# Open3D 工业产品三维视觉检测系统 — 设计文档

## 概述

基于 Open3D 的离线点云检测系统，对涡轮旋转片工件的 PLY 点云进行预处理、配准对齐、尺寸测量和可视化。

**不做参考模型对比**，仅对单工件点云进行分析。

## 架构

```
main.py (入口，编排流程)
  ├── src/preprocessing.py  (数据加载 + 预处理)
  ├── src/registration.py   (PCA 对齐到标准坐标系)
  ├── src/measurement.py    (包围盒 + 尺寸测量 + 截面分析)
  ├── src/visualization.py  (Matplotlib/Open3D 可视化)
  └── src/utils.py          (日志、校验、异常处理)
```

各模块通过 numpy 数组和 Open3D PointCloud 对象传递数据，接口清晰，可独立测试。

## 模块设计

### 1. preprocessing.py
- `load_ply(path: str) -> o3d.geometry.PointCloud` — 读取 PLY，校验有效性
- `voxel_downsample(pcd, voxel_size: float) -> o3d.geometry.PointCloud` — 体素降采样
- `statistical_outlier_removal(pcd, nb_neighbors, std_ratio) -> o3d.geometry.PointCloud` — 统计滤波去噪
- `estimate_normals(pcd, radius, max_nn) -> o3d.geometry.PointCloud` — 法线估计

### 2. registration.py
- `pca_align(pcd) -> o3d.geometry.PointCloud` — PCA 主轴对齐到坐标轴
- `icp_fine_align(source, target, threshold) -> o3d.geometry.PointCloud` — ICP 精配准（多视图拼接时用）

### 3. measurement.py
- `compute_aabb(pcd) -> dict` — 轴对齐包围盒（长、宽、高、中心、角点）
- `compute_obb(pcd) -> dict` — 有向包围盒
- `extract_dimensions(pcd) -> dict` — 提取关键尺寸（AABB 的长宽高）
- `cross_section(pcd, axis, position, thickness) -> np.ndarray` — 截面切片，返回轮廓点
- `deviation_heatmap(pcd) -> np.ndarray` — 基于局部法线一致性生成偏差热力图数据

### 4. visualization.py
- `draw_bounding_box(pcd, bbox, title)` — 包围盒叠加显示
- `draw_dimensions(pcd, dimensions, title)` — 尺寸标注可视化
- `draw_heatmap(pcd, heatmap_data, title)` — 热力图渲染

### 5. utils.py
- `validate_ply(path) -> bool` — 文件格式与完整性校验
- `setup_logging(log_path) -> Logger` — 日志配置
- `PointCloudValidationError`, `FileFormatError` — 自定义异常

## 数据流

```
PLY 文件 → load_ply → validate → voxel_downsample → outlier_removal → estimate_normals
  → pca_align → compute_aabb → extract_dimensions → 可视化输出
```

## 约束

- 不需要相机 SDK，离线处理
- 不涉及参考模型配准对比
- 测量误差目标 < 0.3mm（依赖点云本身精度）
- Python 3.10+，Open3D 0.18+，conda 环境 open3d_pr
