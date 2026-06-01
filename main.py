"""Open3D Industrial 3D Visual Inspection System.-CLI 命令行版本"""
import argparse
import sys
import io
import os
from datetime import datetime

# 强制 stdout/stderr 使用 UTF-8 编码，避免 Windows GBK 终端乱码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.utils import setup_logging, FileFormatError, PointCloudValidationError
from src.preprocessing import load_ply, voxel_downsample, statistical_outlier_removal, estimate_normals
from src.registration import pca_align
from src.measurement import extract_dimensions, cross_section, deviation_heatmap
from src.visualization import draw_bounding_box, draw_heatmap


def main():
    parser = argparse.ArgumentParser(description='三维点云视觉检测系统')
    parser.add_argument('input', help='PLY 点云文件路径')
    parser.add_argument('--voxel-size', type=float, default=0.5, help='体素降采样尺寸')
    parser.add_argument('--name', default=None, help='工件名称（默认使用文件名）')
    parser.add_argument('--output-dir', default='output', help='输出根目录')
    args = parser.parse_args()

    # 工件名称
    workpiece_name = args.name if args.name else os.path.splitext(os.path.basename(args.input))[0]

    # 创建输出子目录: output/<名称>_<时间戳>/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{workpiece_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'inspection.log')
    dim_path = os.path.join(output_dir, 'dimensions.txt')
    bbox_path = os.path.join(output_dir, 'bbox.png')
    heatmap_path = os.path.join(output_dir, 'heatmap.png')

    logger = setup_logging(log_path)
    logger.info("=== 三维检测开始 ===")
    logger.info(f"工件名称: {workpiece_name}")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出目录: {output_dir}")

    try:
        # 1. 加载
        pcd = load_ply(args.input)
        logger.info(f"已加载 {len(pcd.points)} 个点")

        # 2. 预处理
        pcd = voxel_downsample(pcd, voxel_size=args.voxel_size)
        pcd = statistical_outlier_removal(pcd)
        pcd = estimate_normals(pcd)

        # 3. 对齐
        pcd = pca_align(pcd)

        # 4. 测量
        dims = extract_dimensions(pcd)

        # 保存尺寸
        with open(dim_path, 'w', encoding='utf-8') as f:
            f.write(f"工件名称: {workpiece_name}\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {args.input}\n")
            f.write(f"处理点数: {len(pcd.points)}\n")
            f.write(f"---\n")
            f.write(f"长度 (X): {dims['length']:.3f} mm\n")
            f.write(f"宽度 (Y): {dims['width']:.3f} mm\n")
            f.write(f"高度 (Z): {dims['height']:.3f} mm\n")
        logger.info(f"尺寸已保存至 {dim_path}")

        # 截面
        section = cross_section(pcd, axis='z', position=0.0)
        logger.info(f"z=0 处截面: {len(section)} 个点")

        # 偏差热力图
        deviations = deviation_heatmap(pcd)

        # 5. 可视化
        fig_bbox = draw_bounding_box(pcd, dims, title=f"{workpiece_name} — 包围盒")
        fig_bbox.savefig(bbox_path, dpi=150)
        logger.info(f"包围盒图像已保存至 {bbox_path}")

        fig_heat = draw_heatmap(pcd, deviations, title=f"{workpiece_name} — 偏差热力图")
        fig_heat.savefig(heatmap_path, dpi=150)
        logger.info(f"热力图图像已保存至 {heatmap_path}")

        print(f"\n=== 检测完成 ===")
        print(f"工件: {workpiece_name}")
        print(f"尺寸 (mm): 长={dims['length']:.3f} 宽={dims['width']:.3f} 高={dims['height']:.3f}")
        print(f"输出目录: {output_dir}")
        print(f"  {os.path.basename(dim_path)}")
        print(f"  {os.path.basename(bbox_path)}")
        print(f"  {os.path.basename(heatmap_path)}")
        print(f"  {os.path.basename(log_path)}")

    except FileFormatError as e:
        logger.error(str(e))
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except PointCloudValidationError as e:
        logger.error(str(e))
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"未知错误: {e}")
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
