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
        # 仅在控制台可用时添加（exe --windowed 模式下 stdout 不存在）
        import sys as _sys
        if _sys.stdout and _sys.stderr:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(ch)
    return logger


# 模块导入时确保 logger 有兜底 handler（避免 exe 窗口模式报错）
import sys as _sys
if not logging.getLogger("open3d_inspection").handlers:
    if _sys.stdout and _sys.stderr:
        _null = logging.StreamHandler()
    else:
        _null = logging.NullHandler()
    _null.setLevel(logging.WARNING)
    logging.getLogger("open3d_inspection").addHandler(_null)
