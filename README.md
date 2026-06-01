# 3D Inspector — 三维点云视觉检测系统

基于 Open3D + PyQt5 的工业产品三维尺寸检测系统，支持点云预处理、PCA 对齐、包围盒测量、模板对比偏差分析。

## 功能

- **点云预处理**：PLY 加载、体素降采样、统计离群滤波、法线估计
- **自动对齐**：PCA 主轴对齐，将工件旋转到标准坐标系
- **尺寸测量**：AABB 包围盒，自动提取长/宽/高
- **模板对比**（可选）：ICP 配准，逐点偏差距离计算
- **可视化**：3D 点云预览、包围盒标注、偏差热力图
- **批量检测**：一次导入多个文件，一键批量处理，结果独立缓存
- **自动命名匹配**：模板文件按 `文件名_template.ply` 命名规则自动关联

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

1. 点击「导入文件」或「导入文件夹」加载 PLY 文件
2. 在文件列表中点击目标文件
3. 可选：指定标准模板用于偏差对比
4. 点击「开始检测」单件检测，或「一键批量检测」处理全部
5. 切换右侧标签页查看 3D 预览 / 包围盒 / 热力图

### 命令行

```bash
python main.py data/sample_box.ply --voxel-size 1.0 --name "工件A"
```

输出自动保存到 `output/<名称>_<时间戳>/` 目录。

## 模板对比

将标准模板文件放在数据文件同目录，按命名规则自动匹配：

```
data/
├── part_a.ply              # 待检测工件
├── part_a_template.ply     # 自动匹配的模板
├── part_b.ply
└── part_b_ref.ply          # 也支持 _ref 后缀
```

支持的模板后缀：`_template.ply`、`_ref.ply`、`_standard.ply`

也可在 GUI 中通过「指定」按钮手动选择模板。

## 生成样本

```bash
python generate_samples.py
```

会生成 4 个测试几何体：长方体、圆柱体、球体、圆锥体，存放在 `data/` 目录。

## 项目结构

```
open3d_project/
├── main.py                  # CLI 入口
├── run_gui.py               # GUI 入口
├── generate_samples.py      # 样本几何体生成
├── requirements.txt
├── src/
│   ├── utils.py             # 校验、日志、异常
│   ├── preprocessing.py     # PLY 加载、滤波、法线
│   ├── registration.py      # PCA 对齐、ICP
│   ├── measurement.py       # 包围盒、尺寸、偏差
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
