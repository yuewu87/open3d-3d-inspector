import os
import tempfile
import pytest
from src.utils import validate_ply, setup_logging, FileFormatError, PointCloudValidationError


def test_校验合法ply文件通过():
    content = """ply
format ascii 1.0
comment test
element vertex 3
property float x
property float y
property float z
end_header
0 0 0
1 1 1
2 2 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as f:
        f.write(content)
        path = f.name
    try:
        assert validate_ply(path) is True
    finally:
        os.unlink(path)


def test_校验不存在文件返回假():
    assert validate_ply("/nonexistent/file.ply") is False


def test_校验错误后缀返回假():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("not a ply")
        path = f.name
    try:
        assert validate_ply(path) is False
    finally:
        os.unlink(path)


def test_日志配置生成logger并写入():
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
        path = f.name
    try:
        logger = setup_logging(path)
        logger.info("test message")
    finally:
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)
    try:
        with open(path) as lf:
            content = lf.read()
        assert "test message" in content
    finally:
        os.unlink(path)


def test_文件格式异常可抛出():
    with pytest.raises(FileFormatError):
        raise FileFormatError("bad file")


def test_点云校验异常可抛出():
    with pytest.raises(PointCloudValidationError):
        raise PointCloudValidationError("empty point cloud")
