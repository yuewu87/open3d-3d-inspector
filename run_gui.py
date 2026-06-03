"""三维点云视觉检测系统 — GUI 启动入口"""
import sys
import os

# 必须在导入 src 模块之前初始化日志系统
# 否则模块级 logger 在 exe --windowed 模式下会因为 logging.lastResort 绑定的
# stderr.buffer 为 None 而抛出 AttributeError
from src.utils import setup_logging
setup_logging(os.path.join(os.path.dirname(__file__), 'inspection.log'))

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
