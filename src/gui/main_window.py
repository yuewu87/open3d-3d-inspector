"""PyQt5 主窗口 —— 三维点云视觉检测系统 (美化版)"""
import os
import traceback
from datetime import datetime

import numpy as np
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
})

from src.preprocessing import load_ply, voxel_downsample, statistical_outlier_removal, estimate_normals
from src.registration import pca_align
from src.measurement import extract_dimensions, deviation_heatmap
from src.visualization import draw_bounding_box, draw_heatmap
from src.utils import FileFormatError, PointCloudValidationError


# ==================== 样式表 ====================
STYLESHEET = """
/* 全局 */
QMainWindow {
    background-color: #f0f2f5;
}
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

/* 左侧面板 */
#leftPanel {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}
#leftPanel QLabel {
    color: #555555;
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
}
#leftPanel QLineEdit {
    border: 1px solid #dcdfe6;
    border-radius: 6px;
    padding: 7px 10px;
    background: #f5f7fa;
    color: #333;
    font-size: 12px;
}
#leftPanel QLineEdit:focus {
    border-color: #409eff;
    background: #ffffff;
}

/* 按钮 */
QPushButton {
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    border: 1px solid #dcdfe6;
    background: #ffffff;
    color: #606266;
}
QPushButton:hover {
    background: #ecf5ff;
    border-color: #c6e2ff;
    color: #409eff;
}
QPushButton:pressed {
    background: #d9ecff;
}

/* 浏览按钮 */
#browseBtn {
    border-radius: 6px;
    padding: 7px 14px;
    background: #f0f2f5;
}
#browseBtn:hover {
    background: #e6e8eb;
}

/* 运行按钮 */
#runBtn {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #409eff, stop:1 #3a8ee6);
    color: white;
    border: none;
    font-size: 15px;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 8px;
    letter-spacing: 1px;
}
#runBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #66b1ff, stop:1 #409eff);
}
#runBtn:pressed {
    background: #3a8ee6;
}
#runBtn:disabled {
    background: #c0c4cc;
    color: #ffffff;
}

/* 打开目录按钮 */
#openDirBtn {
    background: #f5f7fa;
    border: 1px dashed #dcdfe6;
    color: #909399;
}
#openDirBtn:hover {
    border-color: #409eff;
    color: #409eff;
}

/* 进度条 */
QProgressBar {
    border: none;
    border-radius: 6px;
    background: #e8eaed;
    height: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #409eff, stop:1 #66b1ff);
    border-radius: 6px;
}

/* 结果区域 */
#resultFrame {
    background: #f5f7fa;
    border: 1px solid #e8eaed;
    border-radius: 8px;
    padding: 10px;
}
#resultLabel {
    font-size: 12px;
    color: #909399;
}
#dimValue {
    font-family: "Cascadia Code", "Consolas", "Microsoft YaHei", monospace;
    font-size: 13px;
    color: #303133;
}

/* 分组框 */
QGroupBox {
    font-weight: bold;
    color: #303133;
    border: 1px solid #ebeef5;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #606266;
}

/* 选项卡 */
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background: #f5f7fa;
    border: 1px solid #e0e0e0;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 20px;
    margin-right: 2px;
    color: #909399;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #409eff;
    font-weight: bold;
}
QTabBar::tab:hover {
    color: #409eff;
    background: #ecf5ff;
}

/* 状态栏 */
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e0e0e0;
    color: #909399;
    font-size: 12px;
    padding: 4px;
}

/* 关于对话框 */
QMessageBox {
    background: #ffffff;
}

/* 滚动条 */
QScrollBar:vertical {
    border: none;
    background: #f5f7fa;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #c0c4cc;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #909399;
}
"""


class InspectionWorker(QtCore.QThread):
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
            self.progress.emit("正在加载点云文件...")
            pcd = load_ply(self.filepath)
            self.progress.emit(f"[OK] 已加载 {len(pcd.points):,} 个点")

            self.progress.emit("体素降采样中...")
            pcd = voxel_downsample(pcd, voxel_size=self.voxel_size)

            self.progress.emit("离群点滤波中...")
            pcd = statistical_outlier_removal(pcd)

            self.progress.emit("法线估计中...")
            pcd = estimate_normals(pcd)

            self.progress.emit("PCA 主轴对齐中...")
            pcd = pca_align(pcd)

            self.progress.emit("提取包围盒尺寸...")
            dims = extract_dimensions(pcd)

            self.progress.emit("计算表面偏差...")
            deviations = deviation_heatmap(pcd)

            pts = np.asarray(pcd.points)
            aabb = pcd.get_axis_aligned_bounding_box()
            corners = np.asarray(aabb.get_box_points())

            # 创建输出目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.output_base, f"{self.workpiece_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            dim_path = os.path.join(output_dir, 'dimensions.txt')
            with open(dim_path, 'w', encoding='utf-8') as f:
                f.write(f"工件名称: {self.workpiece_name}\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入文件: {self.filepath}\n")
                f.write(f"处理后点数: {len(pcd.points)}\n")
                f.write(f"---\n")
                f.write(f"长度 (X): {dims['length']:.3f} mm\n")
                f.write(f"宽度 (Y): {dims['width']:.3f} mm\n")
                f.write(f"高度 (Z): {dims['height']:.3f} mm\n")

            self.progress.emit("渲染输出图像...")
            bbox_path = os.path.join(output_dir, 'bbox.png')
            heatmap_path = os.path.join(output_dir, 'heatmap.png')

            fig_bbox = draw_bounding_box(pcd, dims,
                                         title=f"{self.workpiece_name} — Bounding Box")
            fig_bbox.savefig(bbox_path, dpi=150)
            fig_heat = draw_heatmap(pcd, deviations,
                                    title=f"{self.workpiece_name} — Deviation Heatmap")
            fig_heat.savefig(heatmap_path, dpi=150)

            self.progress.emit("[OK] 检测完成")
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
        self.setWindowTitle("3D Inspector — 三维点云视觉检测系统")
        self.resize(1280, 840)
        self.setMinimumSize(960, 600)

        self._setup_menu()
        self._setup_ui()
        self._apply_style()

        self.worker = None
        self.current_results = None

    # ==================== 菜单 ====================
    def _setup_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("QMenuBar { background: #ffffff; border-bottom: 1px solid #e8eaed; }")

        file_menu = menubar.addMenu("  文件(&F)  ")
        file_menu.addAction("打开 PLY 文件...", self._browse_file,
                           QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        file_menu.addSeparator()
        file_menu.addAction("  退出(&Q)", self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        view_menu = menubar.addMenu("  视图(&V)  ")
        view_menu.addAction("重置视图", self._reset_view)

        help_menu = menubar.addMenu("  帮助(&H)  ")
        help_menu.addAction("关于", self._about)

    # ==================== UI 布局 ====================
    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # 使用 QSplitter 实现可拖拽分割
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ===== 左侧面板 =====
        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(380)
        left = QtWidgets.QVBoxLayout(left_panel)
        left.setContentsMargins(16, 16, 16, 16)
        left.setSpacing(4)

        # -- 标题 --
        title = QtWidgets.QLabel("检测控制台")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #303133; padding: 4px 0;")
        left.addWidget(title)

        # -- 文件输入 --
        file_group = QtWidgets.QGroupBox("输入")
        fg = QtWidgets.QVBoxLayout(file_group)
        fg.setSpacing(6)
        file_row = QtWidgets.QHBoxLayout()
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("拖放 PLY 文件到此处...")
        self.file_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("浏览")
        browse_btn.setObjectName("browseBtn")
        browse_btn.setCursor(QtCore.Qt.PointingHandCursor)
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(browse_btn)
        fg.addLayout(file_row)
        left.addWidget(file_group)

        # -- 参数 --
        param_group = QtWidgets.QGroupBox("参数")
        pg = QtWidgets.QVBoxLayout(param_group)
        pg.setSpacing(6)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("工件名称"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("自动识别...")
        name_row.addWidget(self.name_edit, 1)
        pg.addLayout(name_row)

        voxel_row = QtWidgets.QHBoxLayout()
        voxel_row.addWidget(QtWidgets.QLabel("体素尺寸"))
        self.voxel_spin = QtWidgets.QDoubleSpinBox()
        self.voxel_spin.setRange(0.01, 100.0)
        self.voxel_spin.setValue(0.5)
        self.voxel_spin.setDecimals(2)
        self.voxel_spin.setSingleStep(0.1)
        self.voxel_spin.setSuffix(" mm")
        voxel_row.addWidget(self.voxel_spin, 1)
        pg.addLayout(voxel_row)

        left.addWidget(param_group)

        # -- 运行按钮 --
        left.addSpacing(8)
        self.run_btn = QtWidgets.QPushButton("开始检测")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_btn.setMinimumHeight(44)
        self.run_btn.clicked.connect(self._run_inspection)
        left.addWidget(self.run_btn)

        # -- 进度 --
        left.addSpacing(6)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("color: #67c23a; font-size: 12px;")
        left.addWidget(self.progress_label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.hide()
        left.addWidget(self.progress_bar)

        # -- 结果 --
        left.addSpacing(4)
        result_group = QtWidgets.QGroupBox("检测结果")
        rg = QtWidgets.QVBoxLayout(result_group)
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(140)
        self.result_text.setStyleSheet(
            "QTextEdit { background: transparent; border: none; font-family: 'Cascadia Code', "
            "'Consolas', 'Microsoft YaHei', monospace; font-size: 12px; color: #606266; }"
        )
        self.result_text.setPlaceholderText("等待检测...")
        rg.addWidget(self.result_text)
        left.addWidget(result_group)

        # -- 打开目录 --
        self.open_dir_btn = QtWidgets.QPushButton("打开输出目录")
        self.open_dir_btn.setObjectName("openDirBtn")
        self.open_dir_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_dir_btn.clicked.connect(self._open_output_dir)
        self.open_dir_btn.setEnabled(False)
        left.addWidget(self.open_dir_btn)

        left.addStretch()

        # -- 底部版本信息 --
        version_label = QtWidgets.QLabel("v1.0 · Open3D 0.19 + PyQt5")
        version_label.setStyleSheet("color: #c0c4cc; font-size: 11px;")
        version_label.setAlignment(QtCore.Qt.AlignCenter)
        left.addWidget(version_label)

        splitter.addWidget(left_panel)

        # ===== 右侧可视化 =====
        right_panel = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_panel)
        right.setContentsMargins(16, 16, 16, 16)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setDocumentMode(True)

        # Tab 1: 3D 预览
        self.preview_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.preview_canvas, "  3D 预览  ")

        # Tab 2: 包围盒
        self.bbox_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.bbox_canvas, "  包围盒  ")

        # Tab 3: 热力图
        self.heatmap_canvas = FigureCanvas(Figure(figsize=(6, 5)))
        self.tab_widget.addTab(self.heatmap_canvas, "  偏差热力图  ")

        right.addWidget(self.tab_widget)

        # 状态栏
        self._status = QtWidgets.QLabel("就绪 — 请拖放 PLY 文件或点击「浏览」开始检测")
        self._status.setStyleSheet("color: #909399; font-size: 12px; padding: 4px;")
        right.addWidget(self._status)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.setAcceptDrops(True)

        # 显示占位图
        self._show_placeholders()

    # ==================== 样式 ====================
    def _apply_style(self):
        self.setStyleSheet(STYLESHEET)

    # ==================== 占位图 ====================
    def _show_placeholders(self):
        for canvas in [self.preview_canvas, self.bbox_canvas, self.heatmap_canvas]:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "等待检测...\n\n拖放 PLY 文件开始",
                    ha='center', va='center', fontsize=14,
                    color='#c0c4cc', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            canvas.figure.tight_layout()
            canvas.draw()

    # ==================== 拖放 ====================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._status.setText("释放文件以开始检测...")
            self._status.setStyleSheet("color: #409eff; font-size: 12px; padding: 4px;")

    def dragLeaveEvent(self, event):
        self._status.setText("就绪 — 请拖放 PLY 文件或点击「浏览」开始检测")
        self._status.setStyleSheet("color: #909399; font-size: 12px; padding: 4px;")

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.ply'):
                self._set_file(path)
                self._status.setText(f"已载入: {os.path.basename(path)}")
                self._status.setStyleSheet("color: #67c23a; font-size: 12px; padding: 4px;")
                break

    # ==================== 槽函数 ====================
    def _browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 PLY 点云文件", "data",
            "PLY Point Cloud (*.ply);;All Files (*)"
        )
        if path:
            self._set_file(path)
            self._status.setText(f"已载入: {os.path.basename(path)}")
            self._status.setStyleSheet("color: #67c23a; font-size: 12px; padding: 4px;")

    def _set_file(self, path):
        self.file_edit.setText(path)
        if not self.name_edit.text():
            name = os.path.splitext(os.path.basename(path))[0]
            self.name_edit.setText(name)

    def _run_inspection(self):
        filepath = self.file_edit.text().strip()
        if not filepath or not os.path.isfile(filepath):
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择有效的 PLY 文件。")
            return

        name = self.name_edit.text().strip() or os.path.splitext(os.path.basename(filepath))[0]

        self.run_btn.setEnabled(False)
        self.run_btn.setText("处理中...")
        self.progress_bar.show()
        self.result_text.clear()
        self._status.setText("正在检测中...")
        self._status.setStyleSheet("color: #e6a23c; font-size: 12px; padding: 4px;")

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

        self.result_text.setHtml(
            f"<p style='margin:4px 0'><b>工件</b>: {results['workpiece_name']}</p>"
            f"<p style='margin:4px 0'><b>处理点数</b>: {results['point_count']:,}</p>"
            f"<hr style='border:none;border-top:1px solid #e8eaed;margin:8px 0'>"
            f"<p style='margin:4px 0;font-size:14px;color:#303133'>"
            f"<b>L</b>={dims['length']:.2f} mm &nbsp;|&nbsp; "
            f"<b>W</b>={dims['width']:.2f} mm &nbsp;|&nbsp; "
            f"<b>H</b>={dims['height']:.2f} mm</p>"
        )

        self._render_preview(results)
        self._render_bbox(results)
        self._render_heatmap(results)

        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始检测")
        self.progress_bar.hide()
        self.progress_label.setText("")
        self.open_dir_btn.setEnabled(True)
        self._status.setText(f"[OK] 检测完成 — {results['output_dir']}")
        self._status.setStyleSheet("color: #67c23a; font-size: 12px; padding: 4px;")

    def _on_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "检测失败", f"处理过程中发生错误：\n\n{msg}")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始检测")
        self.progress_bar.hide()
        self.progress_label.setText("")
        self._status.setText("[FAIL] 检测失败")
        self._status.setStyleSheet("color: #f56c6c; font-size: 12px; padding: 4px;")

    # ==================== 可视化渲染 ====================
    def _render_preview(self, r):
        canvas = self.preview_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        step = max(1, len(pts) // 5000)
        ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                   s=1, alpha=0.6, c='#409eff')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} — Processed Point Cloud", fontsize=11)
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
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            ax.plot([corners[i,0], corners[j,0]],
                    [corners[i,1], corners[j,1]],
                    [corners[i,2], corners[j,2]], color='#f56c6c', linewidth=2)

        dims = r['dims']
        ax.text2D(0.02, 0.96,
                  f"L={dims['length']:.1f}  W={dims['width']:.1f}  H={dims['height']:.1f}",
                  transform=ax.transAxes, fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbe6',
                           edgecolor='#e6a23c', alpha=0.9))
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} — Bounding Box", fontsize=11)
        canvas.figure.tight_layout(pad=1.5)
        canvas.draw()

    def _render_heatmap(self, r):
        canvas = self.heatmap_canvas
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111, projection='3d')
        pts = r['pts']
        devs = r['deviations']
        step = max(1, len(pts) // 5000)
        sc = ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2],
                        c=devs[::step], s=2, cmap='RdYlGn_r', alpha=0.7)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f"{r['workpiece_name']} — Deviation Heatmap", fontsize=11)
        cbar = canvas.figure.colorbar(sc, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Deviation (deg)', fontsize=9)
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
            self, "关于 — 3D Inspector",
            "<h3>3D Inspector v1.0</h3>"
            "<p>基于 <b>Open3D 0.19</b> + <b>PyQt5</b> 构建</p>"
            "<p>适用于工业产品三维点云尺寸检测</p>"
            "<hr>"
            "<p style='color:#909399;font-size:11px'>"
            "综合实训项目 · 计算机与软件学院</p>"
        )
