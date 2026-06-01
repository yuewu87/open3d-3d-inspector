"""PyQt5 主窗口 —— 三维点云视觉检测系统"""
import os
import traceback
from datetime import datetime

import numpy as np
import open3d as o3d
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.rcParams.update({
    'figure.facecolor': '#fafbfc',
    'axes.facecolor': '#fafbfc',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'font.size': 12,
})

from src.preprocessing import load_ply, voxel_downsample, statistical_outlier_removal, estimate_normals
from src.registration import pca_align, fpfh_ransac_align, icp_fine_align
from src.measurement import extract_dimensions, deviation_heatmap, detect_holes, compute_aabb
from src.visualization import draw_bounding_box, draw_heatmap
from src.utils import FileFormatError, PointCloudValidationError


STYLESHEET = """
QMainWindow { background-color: #f0f2f5; }
QWidget { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; font-size: 15px; }

QLineEdit {
    border: 1px solid #dcdfe6; border-radius: 6px; padding: 8px 10px;
    background: #f5f7fa; color: #333; font-size: 14px;
}
QLineEdit:focus { border-color: #409eff; background: #ffffff; }

#fileList {
    border: none; border-radius: 0;
    background: #ffffff; font-size: 14px;
    outline: none;
}
#fileList::item { padding: 10px 14px; border-bottom: 1px solid #f0f2f5; }
#fileList::item:selected { background: #ecf5ff; color: #409eff; }
#fileList::item:hover { background: #f5f7fa; }

QPushButton {
    border-radius: 6px; padding: 8px 16px; font-size: 14px;
    border: 1px solid #dcdfe6; background: #ffffff; color: #606266;
}
QPushButton:hover { background: #ecf5ff; border-color: #c6e2ff; color: #409eff; }
QPushButton:pressed { background: #d9ecff; }

#runBtn {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #409eff,stop:1 #3a8ee6);
    color: white; border: none; font-size: 15px; font-weight: bold;
    padding: 10px 22px; border-radius: 8px;
}
#runBtn:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #66b1ff,stop:1 #409eff); }
#runBtn:pressed { background: #3a8ee6; }
#runBtn:disabled { background: #c0c4cc; color: #ffffff; }

#openDirBtn {
    background: #f5f7fa; border: 1px dashed #dcdfe6; color: #909399; font-size: 14px;
}
#openDirBtn:hover { border-color: #409eff; color: #409eff; }

QProgressBar { border: none; border-radius: 3px; background: #e8eaed; height: 6px; }
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #409eff,stop:1 #66b1ff);
    border-radius: 3px;
}

QTabWidget::pane { border: 1px solid #e0e0e0; border-radius: 8px; background: #ffffff; }
QTabBar::tab {
    background: #f5f7fa; border: 1px solid #e0e0e0; border-bottom: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    padding: 10px 24px; margin-right: 2px; color: #909399; font-size: 15px;
}
QTabBar::tab:selected { background: #ffffff; color: #409eff; font-weight: bold; }
QTabBar::tab:hover { color: #409eff; background: #ecf5ff; }

QMessageBox { background: #ffffff; }

QScrollBar:vertical { border: none; background: #f5f7fa; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #c0c4cc; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #909399; }
"""


class InspectionWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)
    timeout = QtCore.pyqtSignal(str)   # 模板超时通知

    def __init__(self, filepath, voxel_size, workpiece_name, output_base, template_path=None):
        super().__init__()
        self.filepath = filepath
        self.voxel_size = voxel_size
        self.workpiece_name = workpiece_name
        self.output_base = output_base
        self.template_path = template_path

    def _log(self, msg):
        """写入日志到输出目录"""
        path = getattr(self, '_log_path', None)
        if path:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")
            except Exception:
                pass

    def run(self):
        t_start = datetime.now()
        try:
            self.progress.emit("正在加载点云文件...")
            pcd = load_ply(self.filepath)
            self._log(f"加载完成: {len(pcd.points):,} 点")
            self.progress.emit(f"[OK] 已加载 {len(pcd.points):,} 个点")

            self.progress.emit("体素降采样中...")
            pcd = voxel_downsample(pcd, voxel_size=self.voxel_size)
            self._log(f"体素降采样后: {len(pcd.points):,} 点")
            self.progress.emit("离群点滤波中...")
            pcd = statistical_outlier_removal(pcd)
            self._log(f"离群点滤波后: {len(pcd.points):,} 点")
            self.progress.emit("法线估计中...")
            pcd = estimate_normals(pcd)
            self.progress.emit("PCA 主轴对齐中...")
            pcd = pca_align(pcd)
            self.progress.emit("提取包围盒尺寸...")
            dims = extract_dimensions(pcd)
            self._log(f"AABB: L={dims['length']:.3f} W={dims['width']:.3f} H={dims['height']:.3f} mm")
            self.progress.emit("计算表面偏差...")
            deviations = deviation_heatmap(pcd)

            self.progress.emit("检测孔洞...")
            holes = detect_holes(pcd, axis='z', min_diameter=0.5)

            pts = np.asarray(pcd.points)
            aabb = pcd.get_axis_aligned_bounding_box()
            corners = np.asarray(aabb.get_box_points())

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.output_base, f"{self.workpiece_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            self._log_path = os.path.join(output_dir, 'inspection.log')
            dim_path = os.path.join(output_dir, 'dimensions.txt')
            self._log(f"工件: {self.workpiece_name}")
            self._log(f"文件: {os.path.basename(self.filepath)}")
            self._log(f"体素尺寸: {self.voxel_size} mm")
            has_template = self.template_path and os.path.isfile(self.template_path)
            self._log(f"模板: {'有' if has_template else '无'}")
            icp_rmse = None
            tmpl_deviations = None
            tmpl_pts = None

            # 模板对比: FPFH+RANSAC 粗配准 → ICP 精配准（超时 30s 则跳过）
            TEMPLATE_TIMEOUT = 30
            while has_template:
                if (datetime.now() - t_start).total_seconds() > TEMPLATE_TIMEOUT:
                    self.timeout.emit("模板对比超时(>30s)，跳过模板对比，使用无模板模式")
                    self._log("模板对比超时 (>30s)，跳过")
                    has_template = False
                    break

                self.progress.emit("模板对比 — 加载模板...")
                self._log(f"加载模板: {os.path.basename(self.template_path)}")
                pcd_tmpl = load_ply(self.template_path)
                if (datetime.now() - t_start).total_seconds() > TEMPLATE_TIMEOUT:
                    self._log("模板加载超时，跳过")
                    has_template = False
                    break

                pcd_tmpl = voxel_downsample(pcd_tmpl, voxel_size=self.voxel_size)
                pcd_tmpl = estimate_normals(pcd_tmpl)
                pcd_tmpl = pca_align(pcd_tmpl)

                self.progress.emit("模板对比 — FPFH+RANSAC 粗配准...")
                pcd = fpfh_ransac_align(pcd, pcd_tmpl, voxel_size=self.voxel_size,
                                       distance_threshold=3.0)
                if (datetime.now() - t_start).total_seconds() > TEMPLATE_TIMEOUT:
                    self._log("FPFH+RANSAC 超时，跳过")
                    has_template = False
                    break

                self.progress.emit("模板对比 — ICP 精配准...")
                pcd = icp_fine_align(pcd, pcd_tmpl, threshold=self.voxel_size * 2)
                if (datetime.now() - t_start).total_seconds() > TEMPLATE_TIMEOUT:
                    self._log("ICP 超时，跳过")
                    has_template = False
                    break

                reg_result = o3d.pipelines.registration.evaluate_registration(
                    pcd, pcd_tmpl, self.voxel_size * 2, np.eye(4)
                )
                icp_rmse = reg_result.inlier_rmse
                self._log(f"ICP RMSE: {icp_rmse:.4f} mm")

                self.progress.emit("模板对比 — 计算偏差...")
                tmpl_deviations = np.asarray(pcd.compute_point_cloud_distance(pcd_tmpl))
                self._log(f"模板偏差: max={tmpl_deviations.max():.4f} "
                         f"mean={tmpl_deviations.mean():.4f} std={tmpl_deviations.std():.4f} mm")
                dims = extract_dimensions(pcd)
                break  # 模板处理完成，退出 while 循环

            # 性能计时
            elapsed = (datetime.now() - t_start).total_seconds()
            ms_per_10k = elapsed / max(len(pcd.points), 1) * 10000

            # OBB 有向包围盒
            obb = pcd.get_oriented_bounding_box()
            obb_extent = obb.extent
            obb_dims = {
                'length': float(obb_extent[0]),
                'width': float(obb_extent[1]),
                'height': float(obb_extent[2]),
            }
            self._log(f"OBB: L={obb_dims['length']:.3f} W={obb_dims['width']:.3f} H={obb_dims['height']:.3f} mm")
            self._log(f"耗时: {elapsed:.2f}s ({ms_per_10k:.2f}s/万点)")
            if holes:
                self._log(f"孔洞: {holes}")
            if has_template:
                self._log(f"ICP RMSE: {icp_rmse:.4f} mm")

            with open(dim_path, 'w', encoding='utf-8') as f:
                f.write(f"文件名: {os.path.basename(self.filepath)}\n")
                f.write(f"工件名称: {self.workpiece_name}\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if has_template:
                    f.write(f"模板文件: {os.path.basename(self.template_path)}\n")
                    f.write(f"ICP RMSE: {icp_rmse:.4f} mm\n")
                f.write(f"处理后点数: {len(pcd.points)}\n---\n")
                f.write(f"AABB 轴对齐包围盒 (平行于坐标轴):\n")
                f.write(f"长度 (X): {dims['length']:.3f} mm\n")
                f.write(f"宽度 (Y): {dims['width']:.3f} mm\n")
                f.write(f"高度 (Z): {dims['height']:.3f} mm\n")
                f.write(f"---\nOBB 有向包围盒 (贴合工件方向):\n")
                f.write(f"长度: {obb_dims['length']:.3f} mm\n")
                f.write(f"宽度: {obb_dims['width']:.3f} mm\n")
                f.write(f"高度: {obb_dims['height']:.3f} mm\n")
                f.write(f"---\n")
                f.write(f"处理耗时: {elapsed:.2f} 秒\n")
                f.write(f"性能: {ms_per_10k:.2f} 秒/万点\n")
                if holes:
                    f.write(f"---\n检测到 {len(holes)} 个孔洞:\n")
                    for h_idx, d in enumerate(holes, 1):
                        f.write(f"孔 {h_idx}: 直径 {d:.3f} mm\n")
                if tmpl_deviations is not None:
                    f.write(f"---\n模板对比偏差:\n")
                    f.write(f"最大偏差: {tmpl_deviations.max():.4f} mm\n")
                    f.write(f"平均偏差: {tmpl_deviations.mean():.4f} mm\n")
                    f.write(f"标准差: {tmpl_deviations.std():.4f} mm\n")

            self.progress.emit("渲染输出图像...")
            bbox_path = os.path.join(output_dir, 'bbox.png')
            heatmap_path = os.path.join(output_dir, 'heatmap.png')

            fig_bbox = draw_bounding_box(pcd, dims, title=f"{self.workpiece_name} - Bounding Box")
            fig_bbox.savefig(bbox_path, dpi=150)

            heatmap_data = tmpl_deviations if tmpl_deviations is not None else deviations
            heatmap_title = (f"{self.workpiece_name} vs Template - Deviation"
                             if has_template else f"{self.workpiece_name} - Deviation Heatmap")
            fig_heat = draw_heatmap(pcd, heatmap_data, title=heatmap_title)
            fig_heat.savefig(heatmap_path, dpi=150)

            self.progress.emit("[OK] 检测完成")
            self.finished.emit({
                'dims': dims, 'pts': pts, 'corners': corners,
                'deviations': heatmap_data,
                'bbox_path': bbox_path, 'heatmap_path': heatmap_path,
                'dim_path': dim_path, 'output_dir': output_dir,
                'workpiece_name': self.workpiece_name, 'point_count': len(pcd.points),
                'basename': os.path.basename(self.filepath),
                'has_template': has_template,
                'template_name': os.path.basename(self.template_path) if has_template else '',
                'icp_rmse': icp_rmse,
                'holes': holes,
                'obb': obb_dims, 'elapsed': elapsed, 'ms_per_10k': ms_per_10k,
            })
        except (FileFormatError, PointCloudValidationError) as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Inspector - 三维点云视觉检测系统")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 640)

        self._file_paths = {}          # basename -> full path
        self._templates = {}           # basename -> template path (or None)
        self._all_results = {}         # basename -> results dict
        self._batch_queue = []         # remaining batch items
        self.worker = None
        self.current_results = None
        self._batch_running = False

        self._setup_menu()
        self._setup_ui()
        self._apply_style()
        self.setAcceptDrops(True)
        self._show_placeholders()

    # ==================== 菜单 ====================
    def _setup_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("QMenuBar { background: #ffffff; border-bottom: 1px solid #e8eaed; font-size: 15px; }")

        file_menu = menubar.addMenu("  文件(&F)  ")
        file_menu.addAction("导入 PLY 文件...", self._import_files, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        file_menu.addAction("导入文件夹...", self._import_dir)
        file_menu.addSeparator()
        file_menu.addAction("清除列表", self._clear_file_list)
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        view_menu = menubar.addMenu("  视图(&V)  ")
        view_menu.addAction("重置视图", self._reset_view)

        help_menu = menubar.addMenu("  帮助(&H)  ")
        help_menu.addAction("关于", self._about)

    # ==================== UI 布局 ====================
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ===== 顶部工具栏 =====
        toolbar = QtWidgets.QWidget()
        toolbar.setStyleSheet(
            "background: #ffffff; border-bottom: 1px solid #e0e0e0; padding: 8px 12px;"
        )
        tb = QtWidgets.QHBoxLayout(toolbar)
        tb.setContentsMargins(12, 8, 12, 8)
        tb.setSpacing(10)

        import_btn = QtWidgets.QPushButton("导入文件")
        import_btn.setCursor(QtCore.Qt.PointingHandCursor)
        import_btn.clicked.connect(self._import_files)
        tb.addWidget(import_btn)

        import_dir_btn = QtWidgets.QPushButton("导入文件夹")
        import_dir_btn.setCursor(QtCore.Qt.PointingHandCursor)
        import_dir_btn.clicked.connect(self._import_dir)
        tb.addWidget(import_dir_btn)

        delete_btn = QtWidgets.QPushButton("删除选中")
        delete_btn.setCursor(QtCore.Qt.PointingHandCursor)
        delete_btn.clicked.connect(self._delete_selected)
        tb.addWidget(delete_btn)

        clear_btn = QtWidgets.QPushButton("清空")
        clear_btn.setCursor(QtCore.Qt.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_file_list)
        tb.addWidget(clear_btn)

        tb.addSpacing(20)

        tb.addWidget(QtWidgets.QLabel("体素"))
        self.voxel_spin = QtWidgets.QDoubleSpinBox()
        self.voxel_spin.setRange(0.01, 100.0)
        self.voxel_spin.setValue(0.5)
        self.voxel_spin.setDecimals(2)
        self.voxel_spin.setSingleStep(0.1)
        self.voxel_spin.setSuffix(" mm")
        self.voxel_spin.setFixedWidth(100)
        tb.addWidget(self.voxel_spin)

        tb.addSpacing(8)

        self.run_btn = QtWidgets.QPushButton("开始检测")
        self.run_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_btn.setMinimumHeight(38)
        self.run_btn.clicked.connect(self._run_inspection)
        self.run_btn.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #409eff,stop:1 #3a8ee6); color: white; border: none; "
            "font-size: 15px; font-weight: bold; padding: 10px 22px; border-radius: 8px; }"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #66b1ff,stop:1 #409eff); }"
            "QPushButton:pressed { background: #3a8ee6; }"
            "QPushButton:disabled { background: #c0c4cc; color: #ffffff; }"
        )
        tb.addWidget(self.run_btn)

        self.batch_btn = QtWidgets.QPushButton("一键批量检测")
        self.batch_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.batch_btn.setMinimumHeight(38)
        self.batch_btn.clicked.connect(self._run_batch)
        self.batch_btn.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: bold; "
            "border: 2px solid #409eff; color: #409eff; "
            "border-radius: 8px; padding: 8px 18px; background: #ecf5ff; }"
            "QPushButton:hover { background: #d9ecff; }"
            "QPushButton:disabled { border-color: #c0c4cc; color: #c0c4cc; background: #f5f7fa; }"
        )
        tb.addWidget(self.batch_btn)

        tb.addStretch()

        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("color: #67c23a; font-size: 14px;")
        tb.addWidget(self.progress_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedSize(120, 6)
        self.progress_bar.hide()
        tb.addWidget(self.progress_bar)

        root.addWidget(toolbar)

        # ===== 主体: 水平分割 =====
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # -- 左侧: 文件列表 + 信息 --
        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMinimumWidth(280)
        left = QtWidgets.QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)

        # 文件列表（占据主要空间）
        list_header = QtWidgets.QLabel("  文件列表")
        list_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #303133; "
            "padding: 10px 14px; background: #fafbfc; border-bottom: 1px solid #ebeef5;"
        )
        left.addWidget(list_header)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setAlternatingRowColors(True)
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        left.addWidget(self.file_list, stretch=1)

        # 底部信息区
        info_widget = QtWidgets.QWidget()
        info_widget.setStyleSheet("background: #fafbfc; border-top: 1px solid #ebeef5;")
        info = QtWidgets.QVBoxLayout(info_widget)
        info.setContentsMargins(12, 10, 12, 10)
        info.setSpacing(6)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("工件名称"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("自动识别...")
        name_row.addWidget(self.name_edit, 1)
        self.name_apply_btn = QtWidgets.QPushButton("修改")
        self.name_apply_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.name_apply_btn.setFixedWidth(60)
        self.name_apply_btn.clicked.connect(self._apply_name)
        name_row.addWidget(self.name_apply_btn)
        info.addLayout(name_row)

        # 模板文件
        tmpl_row = QtWidgets.QHBoxLayout()
        tmpl_row.addWidget(QtWidgets.QLabel("标准模板"))
        self.template_label = QtWidgets.QLabel("未匹配")
        self.template_label.setStyleSheet(
            "color: #909399; font-size: 13px; padding: 4px 8px; "
            "background: #ffffff; border: 1px solid #e8eaed; border-radius: 4px;"
        )
        self.template_label.setWordWrap(True)
        tmpl_row.addWidget(self.template_label, 1)
        self.template_btn = QtWidgets.QPushButton("指定")
        self.template_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.template_btn.setFixedWidth(60)
        self.template_btn.clicked.connect(self._browse_template)
        tmpl_row.addWidget(self.template_btn)
        self.template_clear_btn = QtWidgets.QPushButton("清除")
        self.template_clear_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.template_clear_btn.setFixedWidth(60)
        self.template_clear_btn.clicked.connect(self._clear_template)
        tmpl_row.addWidget(self.template_clear_btn)
        info.addLayout(tmpl_row)

        result_label = QtWidgets.QLabel("检测结果")
        result_label.setStyleSheet("font-weight: bold; color: #555;")
        info.addWidget(result_label)
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(100)
        self.result_text.setStyleSheet(
            "QTextEdit { background: #ffffff; border: 1px solid #e8eaed; border-radius: 6px; "
            "font-family: 'Cascadia Code','Consolas','Microsoft YaHei',monospace; "
            "font-size: 14px; color: #606266; padding: 6px; }"
        )
        self.result_text.setPlaceholderText("等待检测...")
        info.addWidget(self.result_text, stretch=1)

        self.open_dir_btn = QtWidgets.QPushButton("打开输出目录")
        self.open_dir_btn.setObjectName("openDirBtn")
        self.open_dir_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_dir_btn.clicked.connect(self._open_output_dir)
        self.open_dir_btn.setEnabled(False)
        info.addWidget(self.open_dir_btn)

        left.addWidget(info_widget)
        splitter.addWidget(left_panel)

        # -- 右侧: 可视化 --
        right_panel = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_panel)
        right.setContentsMargins(16, 16, 16, 16)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setDocumentMode(True)

        self.preview_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.preview_canvas, "  3D 预览  ")
        self.bbox_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.bbox_canvas, "  包围盒  ")
        self.heatmap_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.heatmap_canvas, "  偏差热力图  ")

        right.addWidget(self.tab_widget)

        self._status = QtWidgets.QLabel("就绪 — 请导入 PLY 文件开始检测")
        self._status.setStyleSheet("color: #909399; font-size: 14px; padding: 6px;")
        right.addWidget(self._status)

        splitter.addWidget(right_panel)
        splitter.setSizes([320, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

    # ==================== 样式 ====================
    def _apply_style(self):
        self.setStyleSheet(STYLESHEET)

    # ==================== 占位图 ====================
    def _show_placeholders(self):
        for canvas in [self.preview_canvas, self.bbox_canvas, self.heatmap_canvas]:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "等待检测...\n\n导入 PLY 文件并点击「开始检测」",
                    ha='center', va='center', fontsize=16, color='#c0c4cc',
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            canvas.figure.tight_layout()
            canvas.draw()

    # ==================== 拖放 ====================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._status.setText("释放文件以导入...")
            self._status.setStyleSheet("color: #409eff; font-size: 14px; padding: 6px;")

    def dragLeaveEvent(self, event):
        n = self.file_list.count()
        self._status.setText(f"就绪 — 已导入 {n} 个文件" if n else "就绪 — 请导入 PLY 文件开始检测")
        self._status.setStyleSheet("color: #909399; font-size: 14px; padding: 6px;")

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                paths.extend(
                    os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.ply', '.pcd'))
                )
            elif path.lower().endswith('.ply'):
                paths.append(path)
        if paths:
            self._add_files(paths)

    # ==================== 文件管理 ====================
    def _import_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "导入 PLY 点云文件", "data",
            "Point Cloud (*.ply *.pcd);;PLY (*.ply);;PCD (*.pcd);;All Files (*)"
        )
        if paths:
            self._add_files(paths)

    def _import_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择包含 PLY 文件的文件夹", "data"
        )
        if directory:
            paths = [os.path.join(directory, f) for f in os.listdir(directory)
                     if f.lower().endswith(('.ply', '.pcd'))
                     and not self._is_template_file(f)]
            if paths:
                self._add_files(paths)
            else:
                QtWidgets.QMessageBox.information(self, "提示", "该文件夹中没有 PLY 文件。")

    _TEMPLATE_SUFFIXES = ('_template', '_ref', '_standard')

    @classmethod
    def _is_template_file(cls, basename):
        stem = os.path.splitext(basename)[0]
        return any(stem.endswith(s) for s in cls._TEMPLATE_SUFFIXES)

    def _add_files(self, paths):
        for path in paths:
            basename = os.path.basename(path)
            if self._is_template_file(basename):
                continue  # 跳过模板文件
            self._file_paths[basename] = path
            # 自动匹配模板: 同目录下 <name>_template.ply / <name>_ref.ply / <name>_standard.ply
            dirname = os.path.dirname(path)
            stem = os.path.splitext(basename)[0]
            template = None
            for suffix in ['_template.ply', '_ref.ply', '_standard.ply',
                           '_template.pcd', '_ref.pcd', '_standard.pcd']:
                candidate = os.path.join(dirname, stem + suffix)
                if os.path.isfile(candidate):
                    template = candidate
                    break
            self._templates[basename] = template
        self._rebuild_file_list()
        if paths:
            self.file_list.setCurrentRow(self.file_list.count() - 1)

    def _rebuild_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for basename in sorted(self._file_paths.keys()):
            prefix = "[OK] " if basename in self._all_results else ""
            item = QtWidgets.QListWidgetItem(prefix + basename)
            if basename in self._all_results:
                item.setForeground(QtGui.QColor('#67c23a'))
                item.setFont(QtGui.QFont(self.file_list.font().family(), self.file_list.font().pointSize(), QtGui.QFont.Bold))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def _browse_template(self):
        basename = getattr(self, '_current_basename', '')
        if not basename:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"为 {basename} 选择标准模板", "data",
            "Point Cloud (*.ply *.pcd);;PLY (*.ply);;PCD (*.pcd);;All Files (*)"
        )
        if path:
            self._templates[basename] = path
            self.template_label.setText(os.path.basename(path))
            self.template_label.setStyleSheet(
                "color: #67c23a; font-size: 13px; padding: 4px 8px; "
                "background: #f0f9eb; border: 1px solid #c2e7b0; border-radius: 4px;"
            )

    def _clear_template(self):
        basename = getattr(self, '_current_basename', '')
        if basename and basename in self._templates:
            self._templates[basename] = None
            self.template_label.setText("无 (仅尺寸检测)")
            self.template_label.setStyleSheet(
                "color: #e6a23c; font-size: 13px; padding: 4px 8px; "
                "background: #fdf6ec; border: 1px solid #f5dab1; border-radius: 4px;"
            )

    def _delete_selected(self):
        basename = getattr(self, '_current_basename', '')
        if not basename:
            return
        self._file_paths.pop(basename, None)
        self._templates.pop(basename, None)
        self._all_results.pop(basename, None)
        self._current_basename = ''
        self._rebuild_file_list()
        self.name_edit.clear()
        self._show_placeholders()
        self.result_text.clear()
        n = self.file_list.count()
        self._status.setText(f"已删除: {basename} | 剩余 {n} 个文件" if n else "文件列表为空")
        self._status.setStyleSheet("color: #909399; font-size: 14px; padding: 6px;")
        if n > 0:
            self.file_list.setCurrentRow(0)

    def _clear_file_list(self):
        self._file_paths.clear()
        self._templates.clear()
        self._all_results.clear()
        self.file_list.clear()
        self.name_edit.clear()
        self._current_basename = ''
        self._show_placeholders()
        self.result_text.clear()
        self._status.setText("就绪 — 请导入 PLY 文件开始检测")
        self._status.setStyleSheet("color: #909399; font-size: 14px; padding: 6px;")

    def _on_file_selected(self, current, previous):
        if current is None:
            return
        display = current.text()
        basename = display[5:] if display.startswith("[OK] ") else display
        self._current_basename = basename
        self.name_edit.setText(os.path.splitext(basename)[0])
        # 更新模板显示
        t = self._templates.get(basename)
        if t:
            self.template_label.setText(os.path.basename(t))
            self.template_label.setStyleSheet(
                "color: #67c23a; font-size: 13px; padding: 4px 8px; "
                "background: #f0f9eb; border: 1px solid #c2e7b0; border-radius: 4px;"
            )
        else:
            self.template_label.setText("无 (仅尺寸检测)")
            self.template_label.setStyleSheet(
                "color: #e6a23c; font-size: 13px; padding: 4px 8px; "
                "background: #fdf6ec; border: 1px solid #f5dab1; border-radius: 4px;"
            )

        if basename in self._all_results:
            self._display_results(self._all_results[basename])
            self._status.setText(f"[OK] {basename} - 已有检测结果")
            self._status.setStyleSheet("color: #67c23a; font-size: 14px; padding: 6px;")
        else:
            self._show_placeholders()
            self.result_text.clear()
            self.result_text.setPlaceholderText("等待检测...")
            self._status.setText(f"已选择: {basename} (未检测)")
            self._status.setStyleSheet("color: #909399; font-size: 14px; padding: 6px;")

    def _apply_name(self):
        """手动确认/修改工件名称，同时重写已输出的 txt 文件"""
        basename = getattr(self, '_current_basename', '')
        if not basename:
            return
        new_name = self.name_edit.text().strip()
        if not new_name:
            new_name = os.path.splitext(basename)[0]
            self.name_edit.setText(new_name)
        # 如果已有缓存结果，更新名称并重写 txt
        if basename in self._all_results:
            r = self._all_results[basename]
            r['workpiece_name'] = new_name
            self._display_results(r)
            dim_path = r.get('dim_path', '')
            if dim_path and os.path.exists(dim_path):
                dims = r['dims']
                obb = r.get('obb', {})
                holes = r.get('holes', [])
                elapsed = r.get('elapsed', 0)
                ms_per_10k = r.get('ms_per_10k', 0)
                has_tmpl = r.get('has_template', False)
                tmpl_name = r.get('template_name', '')
                icp_rmse = r.get('icp_rmse')
                devs = r.get('deviations')
                with open(dim_path, 'w', encoding='utf-8') as f:
                    f.write(f"文件名: {basename}\n")
                    f.write(f"工件名称: {new_name}\n")
                    f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if has_tmpl and tmpl_name:
                        f.write(f"模板文件: {tmpl_name}\n")
                    if has_tmpl and icp_rmse is not None:
                        f.write(f"ICP RMSE: {icp_rmse:.4f} mm\n")
                    f.write(f"处理后点数: {r['point_count']}\n---\n")
                    f.write(f"AABB 轴对齐包围盒 (平行于坐标轴):\n")
                    f.write(f"长度 (X): {dims['length']:.3f} mm\n")
                    f.write(f"宽度 (Y): {dims['width']:.3f} mm\n")
                    f.write(f"高度 (Z): {dims['height']:.3f} mm\n")
                    f.write(f"---\nOBB 有向包围盒 (贴合工件方向):\n")
                    f.write(f"长度: {obb['length']:.3f} mm\n")
                    f.write(f"宽度: {obb['width']:.3f} mm\n")
                    f.write(f"高度: {obb['height']:.3f} mm\n")
                    f.write(f"---\n")
                    f.write(f"处理耗时: {elapsed:.2f} 秒\n")
                    f.write(f"性能: {ms_per_10k:.2f} 秒/万点\n")
                    if holes:
                        f.write(f"---\n检测到 {len(holes)} 个孔洞:\n")
                        for h_idx, d in enumerate(holes, 1):
                            f.write(f"孔 {h_idx}: 直径 {d:.3f} mm\n")
                    if has_tmpl and devs is not None:
                        f.write(f"---\n模板对比偏差:\n")
                        f.write(f"最大偏差: {devs.max():.4f} mm\n")
                        f.write(f"平均偏差: {devs.mean():.4f} mm\n")
                        f.write(f"标准差: {devs.std():.4f} mm\n")
                # 追加日志
                log_path = os.path.join(os.path.dirname(dim_path), 'inspection.log')
                if os.path.exists(log_path):
                    with open(log_path, 'a', encoding='utf-8') as lf:
                        lf.write(f"{datetime.now().strftime('%H:%M:%S')} | 工件名称已修改为: {new_name}\n")
        self._status.setText(f"工件名称已更新: {new_name}")
        self._status.setStyleSheet("color: #67c23a; font-size: 14px; padding: 6px;")

    # ==================== 检测 ====================
    def _run_inspection(self, filepath=None, basename=None):
        # QPushButton.clicked 信号会传 False，需要过滤
        if not isinstance(filepath, str) or not filepath:
            basename = getattr(self, '_current_basename', '')
            filepath = self._file_paths.get(basename, '') if basename else ''
        if not filepath or not os.path.isfile(filepath):
            QtWidgets.QMessageBox.warning(self, "提示", "请先从文件列表中选择一个 PLY 文件。")
            return
        if not isinstance(basename, str) or not basename:
            basename = os.path.basename(filepath)

        name = self.name_edit.text().strip() or os.path.splitext(basename)[0]

        self.run_btn.setEnabled(False)
        self.batch_btn.setEnabled(False)
        self.run_btn.setText("处理中...")
        self.progress_bar.show()
        self.result_text.clear()
        self._status.setText(f"正在检测: {basename}...")
        self._status.setStyleSheet("color: #e6a23c; font-size: 14px; padding: 6px;")

        template = self._templates.get(basename) if basename else None
        self.worker = InspectionWorker(filepath, self.voxel_spin.value(), name, 'output', template)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.timeout.connect(self._on_timeout)
        self.worker.start()

    def _on_timeout(self, msg):
        QtWidgets.QMessageBox.warning(self, "模板超时", msg)

    def _on_progress(self, msg):
        self.progress_label.setText(msg)

    def _display_results(self, results):
        """在 UI 中显示检测结果（不触发 worker）"""
        self.current_results = results
        dims = results['dims']
        html = (
            f"<p style='margin:4px 0'><b>工件</b>: {results['workpiece_name']}</p>"
            f"<p style='margin:4px 0'><b>处理点数</b>: {results['point_count']:,} | "
            f"<b>耗时</b>: {results.get('elapsed', 0):.1f}s"
        )
        ms_per_10k = results.get('ms_per_10k', 0)
        perf_color = '#67c23a' if ms_per_10k <= 2.0 else '#e6a23c'
        html += (
            f" (<span style='color:{perf_color}'>{ms_per_10k:.2f}s/万点</span>)</p>"
        )
        if results.get('has_template'):
            html += (
                f"<p style='margin:4px 0'><b>模板对比</b>: "
                f"ICP RMSE = {results.get('icp_rmse', 0):.4f} mm</p>"
            )
        holes = results.get('holes', [])
        if holes:
            hole_str = ', '.join(f'{d:.2f}' for d in holes)
            html += f"<p style='margin:4px 0'><b>孔洞直径</b>: {hole_str} mm</p>"
        html += (
            f"<hr style='border:none;border-top:1px solid #e8eaed;margin:8px 0'>"
            f"<p style='margin:4px 0;font-size:14px;color:#303133'>"
            f"<b>AABB</b> L={dims['length']:.2f} W={dims['width']:.2f} H={dims['height']:.2f} mm</p>"
        )
        obb = results.get('obb', {})
        if obb:
            html += (
                f"<p style='margin:2px 0;font-size:13px;color:#606266'>"
                f"<b>OBB</b> L={obb['length']:.2f} W={obb['width']:.2f} H={obb['height']:.2f} mm</p>"
            )
        if results.get('has_template'):
            html += (
                f"<p style='margin:2px 0;font-size:13px;color:#e6a23c'>"
                f"<b>模板偏差</b> ICP RMSE = {results.get('icp_rmse', 0):.4f} mm</p>"
            )
        self.result_text.setHtml(html)
        self._render_preview(results)
        self._render_bbox(results)
        self._render_heatmap(results)
        self.open_dir_btn.setEnabled(True)

    def _on_finished(self, results):
        # 确定 basename
        basename = results.get('basename', '')
        results['basename'] = basename

        # 存储结果
        self._all_results[basename] = results
        self._rebuild_file_list()

        # 显示结果
        self._display_results(results)

        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始检测")
        self.batch_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_label.setText("")
        self._status.setText(f"[OK] {basename} 检测完成 - {results['output_dir']}")
        self._status.setStyleSheet("color: #67c23a; font-size: 14px; padding: 6px;")

        # 批量模式：处理下一个
        if self._batch_running:
            self._start_next_batch_item()

    def _on_error(self, msg):
        basename = getattr(self, '_current_basename', 'unknown')
        QtWidgets.QMessageBox.critical(self, "检测失败",
                                       f"文件: {basename}\n\n{msg}")
        self.run_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.run_btn.setText("开始检测")
        self.progress_bar.hide()
        self.progress_label.setText("")
        self._status.setText(f"[FAIL] {basename} 检测失败")
        self._status.setStyleSheet("color: #f56c6c; font-size: 14px; padding: 6px;")
        # 批量模式：跳过失败项继续
        if self._batch_running:
            self._start_next_batch_item()

    # ==================== 批量检测 ====================
    def _run_batch(self):
        if not self._file_paths:
            QtWidgets.QMessageBox.warning(self, "提示", "文件列表为空，请先导入 PLY 文件。")
            return
        # 收集未检测的文件
        pending = [(b, p) for b, p in self._file_paths.items() if b not in self._all_results]
        if not pending:
            QtWidgets.QMessageBox.information(self, "提示", "所有文件已完成检测。")
            return

        self._batch_queue = pending
        self._batch_running = True
        self._batch_total = len(pending)
        self._batch_done = 0
        self.run_btn.setEnabled(False)
        self.batch_btn.setEnabled(False)
        self._status.setText(f"批量检测: 0/{self._batch_total} ...")
        self._status.setStyleSheet("color: #409eff; font-size: 14px; padding: 6px;")
        self._start_next_batch_item()

    def _start_next_batch_item(self):
        if not self._batch_queue:
            # 全部完成
            self._batch_running = False
            self.run_btn.setEnabled(True)
            self.batch_btn.setEnabled(True)
            self.progress_bar.hide()
            total = sum(1 for b in self._file_paths if b in self._all_results)
            self._status.setText(f"[OK] 批量检测完成: {total}/{len(self._file_paths)} 个文件")
            self._status.setStyleSheet("color: #67c23a; font-size: 14px; padding: 6px;")
            return

        basename, path = self._batch_queue.pop(0)
        self._batch_done = min(self._batch_total - len(self._batch_queue), self._batch_total)
        self._status.setText(
            f"批量检测: {self._batch_done}/{self._batch_total} - {basename}..."
        )
        self._status.setStyleSheet("color: #409eff; font-size: 14px; padding: 6px;")
        self.progress_label.setText(
            f"进度: {self._batch_done}/{self._batch_total}, 正在处理: {basename}"
        )

        self._run_inspection(filepath=path, basename=basename)

    # ==================== 可视化 ====================
    def _render_preview(self, r):
        canvas = self.preview_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=1, alpha=0.6, c='#409eff')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} - Processed Point Cloud", fontsize=13)
        canvas.figure.tight_layout(pad=1.5)
        canvas.draw()

    def _render_bbox(self, r):
        canvas = self.bbox_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=1, alpha=0.5, c='#c0c4cc')
        corners = r['corners']
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            ax.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]], color='#f56c6c', linewidth=2)
        dims = r['dims']
        ax.text2D(0.02, 0.96, f"L={dims['length']:.1f}  W={dims['width']:.1f}  H={dims['height']:.1f}",
                  transform=ax.transAxes, fontsize=13, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbe6',
                           edgecolor='#e6a23c', alpha=0.9))
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} - Bounding Box", fontsize=13)
        canvas.figure.tight_layout(pad=1.5)
        canvas.draw()

    def _render_heatmap(self, r):
        canvas = self.heatmap_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']; devs = r['deviations']
        step = max(1, len(pts) // 5000)
        has_tmpl = r.get('has_template', False)
        cmap = 'RdYlGn_r' if has_tmpl else 'RdYlGn_r'
        sc = ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                        c=devs[::step], s=2, cmap=cmap, alpha=0.7)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        title = f"{r['workpiece_name']} vs Template" if has_tmpl else f"{r['workpiece_name']} - Deviation"
        ax.set_title(title, fontsize=13)
        cbar = canvas.figure.colorbar(sc, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Deviation (mm)' if has_tmpl else 'Deviation (deg)', fontsize=11)
        canvas.figure.tight_layout(pad=1.5)
        canvas.draw()

    def _open_output_dir(self):
        if self.current_results:
            path = self.current_results['output_dir']
            if os.path.exists(path):
                os.startfile(path)

    def _reset_view(self):
        if self.current_results:
            self._render_preview(self.current_results)
            self._render_bbox(self.current_results)
            self._render_heatmap(self.current_results)
        else:
            self._show_placeholders()

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, "关于 - 3D Inspector",
            "<h3>3D Inspector v1.0</h3>"
            "<p>基于 <b>Open3D 0.19</b> + <b>PyQt5</b> 构建</p>"
            "<p>适用于工业产品三维点云尺寸检测</p>"
            "<hr><p style='color:#909399;font-size:13px'>综合实训项目 - 计算机与软件学院</p>"
        )
