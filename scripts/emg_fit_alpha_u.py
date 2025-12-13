"""
emg_fit_alpha_u.py - PAV 保序回归生成单调 α(u) 门控函数

功能：
1. 从离散的 α* 拟合单调递增的 α(u) 函数
2. 使用 PAV（Pool Adjacent Violators）保序回归算法
3. 输出 α(u) 查表和可视化图表
"""

import os
import sys
import json
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from scipy.stats import isotonic_regression
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体（如果需要）
if HAS_MATPLOTLIB:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def pav_regression(y: np.ndarray) -> np.ndarray:
    """
    实现 PAV（Pool Adjacent Violators）保序回归算法
    
    对于给定的 y（应该已经按x值排序），返回单调递增的 y_isotonic，使得：
    - y_isotonic[i] <= y_isotonic[i+1] for all i
    - 最小化 sum((y_isotonic - y)^2)
    
    Args:
        y: 输入数组（应该已经按对应的x值排序）
    
    Returns:
        单调递增的输出数组
    """
    n = len(y)
    if n <= 1:
        return y.copy()
    
    # 初始化结果数组
    result = y.copy()
    
    # PAV 算法：合并违反单调性的相邻组
    # 使用改进的算法：从右到左扫描，合并违反单调性的组
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result) - 1:
            if result[i] > result[i + 1]:
                # 违反单调性，需要合并相邻的组
                # 找到所有需要合并的连续组
                start = i
                end = i + 1
                
                # 向后查找所有需要合并的值
                while end < len(result) - 1 and result[end] > result[end + 1]:
                    end += 1
                
                # 计算这些值的平均值
                avg = np.mean(result[start:end + 1])
                
                # 赋值
                result[start:end + 1] = avg
                
                changed = True
                i = end + 1
            else:
                i += 1
    
    return result


def fit_alpha_u(
    u_values: np.ndarray,
    alpha_values: np.ndarray,
    use_scipy: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 PAV 保序回归拟合单调递减的 α(u) 函数
    
    注意：根据EMG的物理意义，不确定性越高（u越大），越信任知识（α越小）
    所以 α(u) 应该是单调递减的
    
    Args:
        u_values: 不确定性值数组
        alpha_values: 对应的 α* 值数组
        use_scipy: 是否使用 scipy 的实现（更高效）
    
    Returns:
        (u_fitted, alpha_fitted) - 拟合后的 u 和 α 值
    """
    logger.info("使用 PAV 保序回归拟合 α(u) 函数（单调递减）...")
    
    # 确保 u 值已排序
    sort_indices = np.argsort(u_values)
    u_sorted = u_values[sort_indices]
    alpha_sorted = alpha_values[sort_indices]
    
    # 执行保序回归
    # 由于我们需要单调递减（u增大，α减小），而PAV默认是单调递增
    # 方法：对 -alpha 做单调递增回归，然后取负
    if use_scipy and HAS_SCIPY:
        try:
            alpha_neg = -alpha_sorted
            alpha_fitted_neg = isotonic_regression(alpha_neg, y_min=None, y_max=None, increasing=True)
            alpha_fitted = -alpha_fitted_neg
            
            logger.info("✓ 使用 scipy.stats.isotonic_regression 完成拟合（单调递减）")
        except Exception as e:
            logger.warning(f"scipy 拟合失败: {e}，使用自定义实现")
            # 对 -alpha 做单调递增回归
            alpha_neg = -alpha_sorted
            alpha_fitted_neg = pav_regression(alpha_neg)
            alpha_fitted = -alpha_fitted_neg
    else:
        if not HAS_SCIPY:
            logger.info("未安装 scipy，使用自定义 PAV 实现")
        # 对 -alpha 做单调递增回归
        alpha_neg = -alpha_sorted
        alpha_fitted_neg = pav_regression(alpha_neg)
        alpha_fitted = -alpha_fitted_neg
    
    # 检查单调性（应该是单调递减：alpha[i] >= alpha[i+1]）
    is_monotonic_decreasing = np.all(alpha_fitted[:-1] >= alpha_fitted[1:])
    if not is_monotonic_decreasing:
        logger.warning("⚠ 拟合结果不是单调递减的，可能需要检查输入数据")
        logger.warning(f"  前几个值: {alpha_fitted[:5]}")
    else:
        logger.info("✓ 拟合结果满足单调递减性")
    
    logger.info(f"✓ PAV 回归完成，共 {len(u_sorted)} 个点")
    
    return u_sorted, alpha_fitted


def create_lookup_table(
    u_values: np.ndarray,
    alpha_values: np.ndarray,
    n_points: int = 100
) -> Dict[str, List[float]]:
    """
    创建 α(u) 查表
    
    Args:
        u_values: u 值数组（已排序）
        alpha_values: 对应的 α 值数组
        n_points: 查表的点数
    
    Returns:
        查表字典，包含 u 和 alpha 的列表
    """
    logger.info(f"创建 α(u) 查表，点数: {n_points}")
    
    # 使用线性插值创建均匀的查表
    u_min = u_values.min()
    u_max = u_values.max()
    
    # 创建均匀分布的 u 值
    u_lut = np.linspace(u_min, u_max, n_points)
    
    # 使用线性插值计算对应的 α 值
    alpha_lut = np.interp(u_lut, u_values, alpha_values)
    
    # 确保 α 在 [0, 1] 范围内
    alpha_lut = np.clip(alpha_lut, 0.0, 1.0)
    
    lut = {
        'u': u_lut.tolist(),
        'alpha': alpha_lut.tolist()
    }
    
    logger.info(f"✓ 查表创建完成，u 范围: [{u_min:.4f}, {u_max:.4f}]")
    
    return lut


def plot_alpha_u_curve(
    u_values: np.ndarray,
    alpha_original: np.ndarray,
    alpha_fitted: np.ndarray,
    output_path: str
):
    """
    绘制 α(u) 曲线
    
    Args:
        u_values: u 值数组
        alpha_original: 原始 α* 值
        alpha_fitted: 拟合后的 α 值
        output_path: 输出图片路径
    """
    if not HAS_MATPLOTLIB:
        logger.warning("未安装 matplotlib，跳过图表生成")
        return
    
    logger.info(f"绘制 α(u) 曲线: {output_path}")
    
    plt.figure(figsize=(10, 6))
    
    # 绘制原始数据点
    plt.scatter(u_values, alpha_original, label='原始 α* (每个bucket)', 
                color='blue', alpha=0.6, s=100, zorder=3)
    
    # 绘制拟合曲线
    plt.plot(u_values, alpha_fitted, label='PAV 拟合曲线', 
             color='red', linewidth=2, zorder=2)
    
    plt.xlabel('不确定性 (u)', fontsize=12)
    plt.ylabel('融合权重 α', fontsize=12)
    plt.title('α(u) 门控函数（PAV 保序回归）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 添加说明文字
    plt.text(0.02, 0.98, 
             '说明：不确定性越高，α越小，越信任知识',
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 图表已保存: {output_path}")
    plt.close()


def load_bucket_alpha_star(csv_file: str) -> pd.DataFrame:
    """
    加载 bucket_alpha_star.csv 文件
    
    Args:
        csv_file: CSV 文件路径
    
    Returns:
        DataFrame
    """
    logger.info(f"加载 bucket alpha star 文件: {csv_file}")
    
    if not os.path.exists(csv_file):
        logger.error(f"文件不存在: {csv_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"✓ 加载 {len(df)} 个 bucket")
        return df
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        return pd.DataFrame()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PAV 保序回归生成单调 α(u) 门控函数')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='输入文件（bucket_alpha_star.csv，默认从config读取）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认从config读取）'
    )
    parser.add_argument(
        '--lut-file',
        type=str,
        default=None,
        help='查表输出文件（默认: alpha_u_lut.json）'
    )
    parser.add_argument(
        '--curve-file',
        type=str,
        default=None,
        help='曲线图输出文件（默认: alpha_u_curve.png）'
    )
    parser.add_argument(
        '--lut-points',
        type=int,
        default=100,
        help='查表的点数（默认: 100）'
    )
    parser.add_argument(
        '--use-scipy',
        action='store_true',
        default=True,
        help='使用 scipy 的 isotonic_regression（默认: True）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        config = {}
    
    # 获取参数
    output_dir = args.output_dir or config.get('output_dir', './output')
    
    # 文件路径
    input_file = args.input_file or os.path.join(output_dir, 'bucket_alpha_star.csv')
    lut_file = args.lut_file or os.path.join(output_dir, 'alpha_u_lut.json')
    curve_file = args.curve_file or os.path.join(output_dir, 'alpha_u_curve.png')
    
    logger.info("=" * 60)
    logger.info("PAV 保序回归生成单调 α(u) 门控函数")
    logger.info("=" * 60)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"查表文件: {lut_file}")
    logger.info(f"曲线图: {curve_file}")
    logger.info(f"查表点数: {args.lut_points}")
    logger.info("")
    
    # 加载数据
    df = load_bucket_alpha_star(input_file)
    if len(df) == 0:
        logger.error("无法加载数据文件")
        return 1
    
    # 检查必需的列
    required_cols = ['bucket_id', 'u_mean', 'alpha_star']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"缺少必需的列: {missing_cols}")
        return 1
    
    # 提取数据
    u_values = df['u_mean'].values
    alpha_values = df['alpha_star'].values
    
    logger.info(f"数据点数: {len(u_values)}")
    logger.info(f"u 范围: [{u_values.min():.4f}, {u_values.max():.4f}]")
    logger.info(f"α 范围: [{alpha_values.min():.4f}, {alpha_values.max():.4f}]")
    logger.info("")
    
    # 拟合 α(u) 函数
    u_fitted, alpha_fitted = fit_alpha_u(u_values, alpha_values, use_scipy=args.use_scipy)
    
    logger.info("")
    logger.info("拟合结果:")
    for i in range(len(u_fitted)):
        logger.info(f"  u={u_fitted[i]:.4f}: α={alpha_fitted[i]:.4f} "
                   f"(原始: {alpha_values[np.argsort(u_values)][i]:.4f})")
    logger.info("")
    
    # 创建查表
    lut = create_lookup_table(u_fitted, alpha_fitted, n_points=args.lut_points)
    
    # 保存查表
    os.makedirs(os.path.dirname(lut_file) if os.path.dirname(lut_file) else '.', exist_ok=True)
    with open(lut_file, 'w', encoding='utf-8') as f:
        json.dump(lut, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 查表已保存: {lut_file}")
    
    # 绘制曲线
    plot_alpha_u_curve(
        u_fitted,
        alpha_values[np.argsort(u_values)],
        alpha_fitted,
        curve_file
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("完成")
    logger.info("=" * 60)
    logger.info(f"输出文件:")
    logger.info(f"  - 查表: {lut_file}")
    logger.info(f"  - 曲线图: {curve_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

