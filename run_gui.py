"""三维点云视觉检测系统 — GUI 启动入口"""
import sys, os
# --windowed 模式兜底：stderr 可能不可用，重定向到 nul
if getattr(sys.stderr, 'buffer', None) is None:
    sys.stderr = open(os.devnull, 'w')
if getattr(sys.stdout, 'buffer', None) is None:
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
