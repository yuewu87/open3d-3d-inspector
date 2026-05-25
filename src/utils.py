import os
import logging


class FileFormatError(Exception):
    """文件格式错误异常"""
    pass


class PointCloudValidationError(Exception):
    """点云校验失败异常"""
    pass


def validate_ply(filepath: str) -> bool:
    """校验 PLY 文件路径是否有效"""
    if not os.path.isfile(filepath):
        return False
    if not filepath.lower().endswith('.ply'):
        return False
    return True


def setup_logging(log_path: str = "inspection.log") -> logging.Logger:
    """配置日志系统，同时输出到文件和控制台"""
    logger = logging.getLogger("open3d_inspection")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(ch)
    return logger
