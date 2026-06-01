# 3D Inspector — 三维点云视觉检测系统

基于 Open3D + PyQt5 的工业产品三维尺寸检测系统，支持 PLY/PCD 点云预处理、FPFH+RANSAC+ICP 配准、AABB/OBB 包围盒、孔径测量、模板对比偏差分析。

## 功能

- **点云预处理**：PLY/PCD 加载、体素降采样、统计离群滤波、法线估计
- **自动对齐**：PCA 主轴对齐，将工件旋转到标准坐标系
- **配准对比**：FPFH+RANSAC 粗配准 + ICP 精配准（RMSE ≤ 0.5mm），支持标准模板偏差分析
- **尺寸测量**：AABB + OBB 包围盒，自动提取长/宽/高
- **孔径检测**：截面圆孔检测，自动计算孔径
- **偏差热力图**：逐点偏差距离可视化
- **性能监控**：处理耗时实时显示（目标 ≤ 2s/万点）
- **批量检测**：一次导入多个文件，一键批量处理，结果独立缓存
- **模板智能过滤**：导入文件时自动跳过模板文件（`_template` / `_ref` / `_standard` 后缀）
- **超时保护**：模板对比超过 30 秒自动跳过，写入日志后继续无模板检测

## 环境

- Python 3.10+
- Open3D 0.19+
- PyQt5 5.15+
- NumPy、Matplotlib

conda 环境：

```bash
conda create -n open3d_pr python=3.10
conda activate open3d_pr
pip install -r requirements.txt
```

## 快速开始

### GUI（推荐）

```bash
python run_gui.py
```

1. 点击「导入文件」或「导入文件夹」加载 PLY/PCD 文件
2. 在文件列表中点击目标文件
3. 可选：指定标准模板用于偏差对比
4. 点击「开始检测」单件检测，或「一键批量检测」处理全部
5. 切换右侧标签页查看 3D 预览 / 包围盒 / 偏差热力图

### 命令行

```bash
python main.py data/01.ply --voxel-size 1.0 --name "工件A"
```

输出自动保存到 `output/<名称>_<时间戳>/` 目录。

## 模板对比

将标准模板文件放在数据文件同目录，按命名规则自动匹配：

```
data/
├── part_a.ply              # 待检测工件
├── part_a_template.ply     # 自动匹配的模板
├── part_b.ply
└── part_b_ref.pcd          # 也支持 PCD 格式
```

支持的模板后缀：`_template.ply` / `_ref.ply` / `_standard.ply` / `_template.pcd` / `_ref.pcd` / `_standard.pcd`

也可在 GUI 中通过「指定」按钮手动选择模板。

## 检测输出

```
output/<名称>_<时间戳>/
├── dimensions.txt   # 尺寸、OBB、孔径、性能数据
├── bbox.png         # 包围盒可视化
├── heatmap.png      # 偏差热力图
└── inspection.log   # 处理日志
```

## 项目结构

```
open3d_project/
├── main.py                  # CLI 入口
├── run_gui.py               # GUI 入口
├── requirements.txt
├── src/
│   ├── utils.py             # 校验、日志、异常 (PLY/PCD)
│   ├── preprocessing.py     # 点云加载、滤波、法线
│   ├── registration.py      # PCA、FPFH+RANSAC、ICP
│   ├── measurement.py       # AABB/OBB、孔径检测、偏差
│   ├── visualization.py     # matplotlib 渲染
│   └── gui/
│       └── main_window.py   # PyQt5 主窗口
├── tests/                   # 单元测试 (pytest)
├── data/                    # 点云数据
├── output/                  # 检测结果（gitignore）
└── docs/                    # 设计文档
```

## 运行测试

```bash
pytest tests/ -v
```
