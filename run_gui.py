"""三维点云视觉检测系统 — GUI 启动入口"""
import sys
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
