"""Open3D Industrial 3D Visual Inspection System."""
import argparse
import sys
import io

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
    parser.add_argument('--voxel-size', type=float, default=0.001, help='体素降采样尺寸 (米)')
    parser.add_argument('--output-dim', default='dimensions.txt', help='尺寸输出文件')
    parser.add_argument('--output-bbox', default='bbox.png', help='包围盒输出图像')
    parser.add_argument('--output-heatmap', default='heatmap.png', help='热力图输出图像')
    parser.add_argument('--log', default='inspection.log', help='日志文件路径')
    args = parser.parse_args()

    logger = setup_logging(args.log)
    logger.info("=== 三维检测开始 ===")
    logger.info(f"输入文件: {args.input}")

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
        with open(args.output_dim, 'w') as f:
            f.write(f"长度 (X): {dims['length']*1000:.3f} mm\n")
            f.write(f"宽度 (Y): {dims['width']*1000:.3f} mm\n")
            f.write(f"高度 (Z): {dims['height']*1000:.3f} mm\n")
        logger.info(f"尺寸已保存至 {args.output_dim}")

        # 截面示例（z=0 处）
        section = cross_section(pcd, axis='z', position=0.0)
        logger.info(f"z=0 处截面: {len(section)} 个点")

        # 偏差热力图
        deviations = deviation_heatmap(pcd)

        # 5. 可视化
        fig_bbox = draw_bounding_box(pcd, dims, title="涡轮旋转片 — 包围盒")
        fig_bbox.savefig(args.output_bbox, dpi=150)
        logger.info(f"包围盒图像已保存至 {args.output_bbox}")

        fig_heat = draw_heatmap(pcd, deviations, title="涡轮旋转片 — 偏差热力图")
        fig_heat.savefig(args.output_heatmap, dpi=150)
        logger.info(f"热力图图像已保存至 {args.output_heatmap}")

        print(f"\n=== 检测完成 ===")
        print(f"尺寸 (mm): 长={dims['length']*1000:.3f} 宽={dims['width']*1000:.3f} 高={dims['height']*1000:.3f}")
        print(f"输出文件: {args.output_dim}, {args.output_bbox}, {args.output_heatmap}")

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
