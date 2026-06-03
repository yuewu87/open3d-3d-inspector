import os
import logging


class FileFormatError(Exception):
    """文件格式错误异常"""
    pass


class PointCloudValidationError(Exception):
    """点云校验失败异常"""
    pass


SUPPORTED_FORMATS = ('.ply', '.pcd')


def validate_ply(filepath: str) -> bool:
    """校验点云文件路径（支持 PLY/PCD）"""
    if not os.path.isfile(filepath):
        return False
    ext = os.path.splitext(filepath.lower())[1]
    return ext in SUPPORTED_FORMATS


def validate_point_cloud(filepath: str) -> bool:
    """校验点云文件路径是否有效（推荐使用）"""
    return validate_ply(filepath)


def setup_logging(log_path: str = "inspection.log") -> logging.Logger:
    """配置日志系统，输出到文件"""
    logger = logging.getLogger("open3d_inspection")
    logger.setLevel(logging.INFO)
    # 清除已有 handler，重新创建文件 handler
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)
    return logger


# 模块导入时添加 NullHandler 兜底（防止 PyInstaller --windowed 模式下 logging.lastResort 崩溃）
# lastResort 指向 sys.stderr，但 windowed 模式下 stderr.buffer 为 None，写入会抛 AttributeError
_null_h = logging.NullHandler()
_null_h.setLevel(logging.DEBUG)
logging.getLogger("open3d_inspection").addHandler(_null_h)
