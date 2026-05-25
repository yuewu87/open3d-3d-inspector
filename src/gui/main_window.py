"""PyQt5 主窗口 —— 三维点云视觉检测系统"""
import os
import traceback
from datetime import datetime

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.preprocessing import load_ply, voxel_downsample, statistical_outlier_removal, estimate_normals
from src.registration import pca_align
from src.measurement import extract_dimensions, deviation_heatmap
from src.visualization import draw_bounding_box, draw_heatmap
from src.utils import FileFormatError, PointCloudValidationError


class InspectionWorker(QtCore.QThread):
    """后台处理线程 —— 只做数据处理，不操作 matplotlib"""
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, filepath, voxel_size, workpiece_name, output_base):
        super().__init__()
        self.filepath = filepath
        self.voxel_size = voxel_size
        self.workpiece_name = workpiece_name
        self.output_base = output_base

    def run(self):
        try:
            self.progress.emit("加载点云...")
            pcd = load_ply(self.filepath)
            self.progress.emit(f"已加载 {len(pcd.points)} 个点")

            self.progress.emit("体素降采样...")
            pcd = voxel_downsample(pcd, voxel_size=self.voxel_size)

            self.progress.emit("离群点滤波...")
            pcd = statistical_outlier_removal(pcd)

            self.progress.emit("法线估计...")
            pcd = estimate_normals(pcd)

            self.progress.emit("PCA 对齐...")
            pcd = pca_align(pcd)

            self.progress.emit("提取尺寸...")
            dims = extract_dimensions(pcd)

            self.progress.emit("计算偏差...")
            deviations = deviation_heatmap(pcd)

            # 提取点云数据（线程安全）
            pts = np.asarray(pcd.points)
            aabb = pcd.get_axis_aligned_bounding_box()
            corners = np.asarray(aabb.get_box_points())

            # 创建输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.output_base, f"{self.workpiece_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            # 保存尺寸
            dim_path = os.path.join(output_dir, 'dimensions.txt')
            with open(dim_path, 'w', encoding='utf-8') as f:
                f.write(f"工件名称: {self.workpiece_name}\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入文件: {self.filepath}\n")
                f.write(f"处理点数: {len(pcd.points)}\n")
                f.write(f"---\n")
                f.write(f"长度 (X): {dims['length']:.3f} mm\n")
                f.write(f"宽度 (Y): {dims['width']:.3f} mm\n")
                f.write(f"高度 (Z): {dims['height']:.3f} mm\n")

            # 保存图片
            self.progress.emit("渲染图像...")
            bbox_path = os.path.join(output_dir, 'bbox.png')
            heatmap_path = os.path.join(output_dir, 'heatmap.png')

            fig_bbox = draw_bounding_box(pcd, dims, title=f"{self.workpiece_name} — 包围盒")
            fig_bbox.savefig(bbox_path, dpi=150)

            fig_heat = draw_heatmap(pcd, deviations, title=f"{self.workpiece_name} — 偏差热力图")
            fig_heat.savefig(heatmap_path, dpi=150)

            # 保存实际渲染用的 figure 引用（matplotlib 在 backend_agg 下安全）
            self.progress.emit("完成!")
            self.finished.emit({
                'dims': dims,
                'pts': pts,
                'corners': corners,
                'deviations': deviations,
                'bbox_path': bbox_path,
                'heatmap_path': heatmap_path,
                'dim_path': dim_path,
                'output_dir': output_dir,
                'workpiece_name': self.workpiece_name,
                'point_count': len(pcd.points),
            })

        except (FileFormatError, PointCloudValidationError) as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维点云视觉检测系统")
        self.resize(1200, 800)

        self._setup_menu()
        self._setup_ui()
        self._setup_style()

        self.worker = None
        self.current_results = None

    # ==================== 菜单 ====================
    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction("打开 PLY...", self._browse_file, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction("关于", self._about)

    # ==================== UI 布局 ====================
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # ----- 左侧控制面板 -----
        left_panel = QtWidgets.QWidget()
        left_panel.setMaximumWidth(320)
        left = QtWidgets.QVBoxLayout(left_panel)

        left.addWidget(QtWidgets.QLabel("PLY 文件:"))
        file_row = QtWidgets.QHBoxLayout()
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("选择或拖入 PLY 文件...")
        self.file_edit.setReadOnly(True)
        btn = QtWidgets.QPushButton("浏览...")
        btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(btn)
        left.addLayout(file_row)

        self.setAcceptDrops(True)

        left.addWidget(QtWidgets.QLabel("工件名称:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("自动使用文件名")
        left.addWidget(self.name_edit)

        left.addWidget(QtWidgets.QLabel("体素降采样尺寸:"))
        voxel_row = QtWidgets.QHBoxLayout()
        self.voxel_spin = QtWidgets.QDoubleSpinBox()
        self.voxel_spin.setRange(0.01, 100.0)
        self.voxel_spin.setValue(0.5)
        self.voxel_spin.setDecimals(3)
        self.voxel_spin.setSingleStep(0.1)
        voxel_row.addWidget(self.voxel_spin)
        voxel_row.addStretch()
        left.addLayout(voxel_row)

        left.addSpacing(10)
        self.run_btn = QtWidgets.QPushButton("开始检测")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run_inspection)
        self.run_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; }")
        left.addWidget(self.run_btn)

        left.addSpacing(10)
        self.progress_label = QtWidgets.QLabel("就绪")
        left.addWidget(self.progress_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        left.addWidget(self.progress_bar)

        left.addSpacing(5)
        left.addWidget(QtWidgets.QLabel("检测结果:"))
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(160)
        self.result_text.setStyleSheet("font-family: 'Consolas', 'Microsoft YaHei'; font-size: 12px;")
        left.addWidget(self.result_text)

        self.open_dir_btn = QtWidgets.QPushButton("打开输出目录")
        self.open_dir_btn.clicked.connect(self._open_output_dir)
        self.open_dir_btn.setEnabled(False)
        left.addWidget(self.open_dir_btn)

        left.addStretch()
        main_layout.addWidget(left_panel)

        # ----- 右侧可视化面板 -----
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget, stretch=1)

        # Tab 1: 3D 预览
        self.preview_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.preview_canvas, "3D 预览")

        # Tab 2: 包围盒
        self.bbox_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.bbox_canvas, "包围盒")

        # Tab 3: 热力图
        self.heatmap_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.heatmap_canvas, "偏差热力图")

        self.statusBar().showMessage("就绪 — 请选择 PLY 文件开始")

    def _setup_style(self):
        self.setStyleSheet("QMainWindow { background-color: #f5f5f5; }")

    # ==================== 拖放 ====================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.ply'):
                self._set_file(path)
                break

    # ==================== 槽函数 ====================
    def _browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 PLY 点云文件", "data", "PLY Files (*.ply);;All Files (*)"
        )
        if path:
            self._set_file(path)

    def _set_file(self, path):
        self.file_edit.setText(path)
        if not self.name_edit.text():
            name = os.path.splitext(os.path.basename(path))[0]
            self.name_edit.setText(name)

    def _run_inspection(self):
        filepath = self.file_edit.text().strip()
        if not filepath or not os.path.isfile(filepath):
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择有效的 PLY 文件。")
            return

        name = self.name_edit.text().strip() or os.path.splitext(os.path.basename(filepath))[0]

        self.run_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_label.setText("处理中...")
        self.result_text.clear()
        self.statusBar().showMessage("正在检测...")

        self.worker = InspectionWorker(filepath, self.voxel_spin.value(), name, 'output')
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg):
        self.progress_label.setText(msg)

    def _on_finished(self, results):
        self.current_results = results
        dims = results['dims']

        self.result_text.setText(
            f"工件: {results['workpiece_name']}\n"
            f"处理后点数: {results['point_count']}\n"
            f"───\n"
            f"长度 (X): {dims['length']:.3f} mm\n"
            f"宽度 (Y): {dims['width']:.3f} mm\n"
            f"高度 (Z): {dims['height']:.3f} mm"
        )

        # 在主线程中渲染所有可视化
        self._render_preview(results)
        self._render_bbox(results)
        self._render_heatmap(results)

        self.run_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_label.setText("检测完成")
        self.open_dir_btn.setEnabled(True)
        self.statusBar().showMessage(f"检测完成 — {results['output_dir']}")

    def _on_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "检测失败", msg)
        self.run_btn.setEnabled(True)
        self.progress_bar.hide()
        self.progress_label.setText("检测失败")

    # ==================== 可视化渲染 ====================
    def _render_preview(self, r):
        """3D 点云预览"""
        canvas = self.preview_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, alpha=0.6, c='steelblue')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} — 处理后点云")
        canvas.figure.tight_layout()
        canvas.draw()

    def _render_bbox(self, r):
        """包围盒可视化"""
        canvas = self.bbox_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2], s=1, alpha=0.5, c='steelblue')

        corners = r['corners']
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]], 'r-', linewidth=2)

        dims = r['dims']
        ax.text2D(0.02, 0.98,
                  f"长: {dims['length']:.2f} mm\n宽: {dims['width']:.2f} mm\n高: {dims['height']:.2f} mm",
                  transform=ax.transAxes, fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} — 包围盒")
        canvas.figure.tight_layout()
        canvas.draw()

    def _render_heatmap(self, r):
        """偏差热力图"""
        canvas = self.heatmap_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        devs = r['deviations']
        step = max(1, len(pts) // 5000)
        sc = ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                        c=devs[::step], s=2, cmap='jet', alpha=0.7)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} — 偏差热力图")
        cbar = canvas.figure.colorbar(sc, ax=ax, shrink=0.5)
        cbar.set_label('偏差 (度)')
        canvas.figure.tight_layout()
        canvas.draw()

    def _open_output_dir(self):
        if self.current_results:
            path = self.current_results['output_dir']
            if os.path.exists(path):
                os.startfile(path)

    def _about(self):
        QtWidgets.QMessageBox.about(
            self, "关于",
            "三维点云视觉检测系统 v1.0\n\n基于 Open3D + PyQt5\n适用于工业产品三维尺寸检测"
        )
