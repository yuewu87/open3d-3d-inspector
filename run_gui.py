"""三维点云视觉检测系统 — GUI 启动入口"""
import sys
import os

# PyInstaller --windowed 模式下 stderr/stdout 的 buffer 为 None，
# 任何模块（含 Open3D）的 logging/print 写入都会触发 AttributeError。
# 在导入任何模块之前重定向到 nul，杜绝崩溃。
if sys.stderr is None or getattr(sys.stderr, 'buffer', True) is None:
    sys.stderr = open(os.devnull, 'w')
if sys.stdout is None or getattr(sys.stdout, 'buffer', True) is None:
    sys.stdout = open(os.devnull, 'w')

from PyQt5 import QtWidgets
from src.gui.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
