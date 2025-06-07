import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_runs(data, max_lpips_thresh=None):
    """
    对加载的JSON数据进行处理和分析。
    """
    if not data:
        print("错误：JSON文件中没有数据。")
        return None, None

    # 数据预处理
    df = pd.DataFrame(data)
    df = df.dropna(subset=['coefficients'])
    if df.empty:
        print("错误：未找到包含'coefficients'的有效运行记录。")
        return None, None

    if max_lpips_thresh is not None and max_lpips_thresh > 0:
        if 'lpips_distance' in df.columns:
            print("\n" + "-"*15 + f" 应用质量门槛 (LPIPS <= {max_lpips_thresh}) " + "-"*15)
            original_count = len(df)
            df = df.dropna(subset=['lpips_distance'])
            df = df[df['lpips_distance'] <= max_lpips_thresh]
            filtered_count = len(df)
            print(f"已从 {original_count} 条记录中筛选出 {filtered_count} 条符合条件的记录。")
            if df.empty:
                print("警告：应用阈值后，没有符合条件的运行记录。")
                return None, None
        else:
            print("警告: 数据中不包含 'lpips_distance'，无法应用质量门槛。")
    
    df['hit_ratio'] = (df['cache_hits'] / df['total_inferences']).fillna(0)

    # 寻找基准和计算节省时间 (仅用于“速度-命中率”得分)
    baseline_runs = df[df['rel_l1_thresh'] == 0]
    if baseline_runs.empty:
        print("警告：未找到基准运行 (rel_l1_thresh == 0)，无法计算'节省时间'相关的得分。")
        baseline_time = df['generation_time'].max()
    else:
        baseline_time = baseline_runs['generation_time'].min()
    
    print(f"基准运行时间 (最快无缓存): {baseline_time:.2f} 秒")
    
    df['time_saved'] = baseline_time - df['generation_time']
    df.loc[df['time_saved'] < 0, 'time_saved'] = 0

    if 'lpips_distance' in df.columns:
        # 命中率 / LPIPS距离
        df['score_lpips'] = (df['hit_ratio'] / df['lpips_distance'].replace(0, float('inf'))).fillna(0)
    else:
        print("警告: 数据中不包含 'lpips_distance'，跳过LPIPS相关分析。")
        df['score_lpips'] = 0

    # “速度-命中率”的得分公式保持不变
    df['score_hit_ratio'] = df['time_saved'] * df['hit_ratio']

    # 找出各项最佳参数
    results = {}
    cached_runs = df[df['rel_l1_thresh'] > 0]
    if cached_runs.empty:
        print("警告: 未找到任何开启缓存的运行记录，无法推荐最优参数。")
        return df, None

    best_indices = {
        "最快生成速度": cached_runs['generation_time'].idxmin(),
        "最高缓存命中率": cached_runs['hit_ratio'].idxmax(),
        "最佳速度-命中率综合效率": cached_runs['score_hit_ratio'].idxmax(), # 名字更新
    }
    if df['score_lpips'].sum() > 0 and 'lpips_distance' in cached_runs.columns and cached_runs['lpips_distance'].notna().any():
        best_indices["最低LPIPS (最佳画质)"] = cached_runs['lpips_distance'].idxmin()
        best_indices["最佳质量-命中率综合效率 (LPIPS)"] = cached_runs['score_lpips'].idxmax() # 名字更新

    for name, idx in best_indices.items():
        best_run = df.loc[idx]
        results[name] = {
            "coefficients": best_run['coefficients'],
            "value": {
                "生成时间": f"{best_run['generation_time']:.2f}s",
                "命中率": f"{best_run['hit_ratio']:.2%}",
                "LPIPS": f"{best_run.get('lpips_distance', 'N/A'):.4f}" if pd.notna(best_run.get('lpips_distance')) else "N/A",
                "速度-命中率得分": f"{best_run['score_hit_ratio']:.2f}",
                "质量-命中率得分(LPIPS)": f"{best_run['score_lpips']:.4f}", # 增加小数位精度
            }
        }
        
    return df, results

def print_results(results):
    if not results:
        return
        
    print("\n" + "="*25 + " 分析结果 " + "="*25)
    for name, data in results.items():
        print(f"\n--- {name} ---")
        print(f"  🏆 最佳Coefficients: {data['coefficients']}")
        print("     相关指标:")
        for key, val in data['value'].items():
            print(f"       - {key}: {val}")
    print("\n" + "="*62)


def create_plots(df, results, max_lpips_thresh=None):
    if df is None or results is None:
        print("Insufficient data to generate plots.")
        return

    plot_df = df[df['rel_l1_thresh'] > 0].copy()
    if plot_df.empty:
        print("No cache-enabled run data available, cannot generate plots.")
        return

    # Plot 1: Time vs LPIPS
    if 'score_lpips' in plot_df.columns and plot_df['lpips_distance'].notna().any():
        plt.style.use('seaborn-v0_8-darkgrid')
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        scatter1 = ax1.scatter(plot_df['generation_time'], plot_df['lpips_distance'], 
                               c=plot_df['score_lpips'], cmap='viridis', alpha=0.7, s=50)
        fig1.colorbar(scatter1, label='Quality-Hit Ratio Score (Hit Ratio / LPIPS)')

        title = 'Generation Speed vs Image Quality (LPIPS)'
        if max_lpips_thresh:
            title += f'\n(Filtering applied, LPIPS <= {max_lpips_thresh})'
        ax1.set_title(title, fontsize=16)

        ax1.set_xlabel('Generation Time (seconds) - Lower is Better', fontsize=12)
        ax1.set_ylabel('LPIPS Distance - Lower is Better', fontsize=12)

        best_lpips_score_idx = plot_df['score_lpips'].idxmax()
        best_point = plot_df.loc[best_lpips_score_idx]
        ax1.scatter(best_point['generation_time'], best_point['lpips_distance'], 
                    color='red', s=150, edgecolor='black', marker='*', 
                    label='Best Quality-Hit Ratio Point')
        ax1.text(best_point['generation_time'], best_point['lpips_distance'], 
                 '  Best Combined Point', color='red', ha='left')
        ax1.legend()
        ax1.grid(True)

    # Plot 2: Time vs Cache Hit Ratio
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    scatter2 = ax2.scatter(plot_df['generation_time'], plot_df['hit_ratio'], 
                           c=plot_df['score_hit_ratio'], cmap='plasma', alpha=0.7, s=50)
    fig2.colorbar(scatter2, label='Speed-Hit Ratio Score (Time Saved * Hit Ratio)')
    ax2.set_title('Generation Speed vs Cache Hit Rate', fontsize=16)
    ax2.set_xlabel('Generation Time (seconds) - Lower is Better', fontsize=12)
    ax2.set_ylabel('Cache Hit Rate - Higher is Better', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    best_hr_score_idx = plot_df['score_hit_ratio'].idxmax()
    best_point_hr = plot_df.loc[best_hr_score_idx]
    ax2.scatter(best_point_hr['generation_time'], best_point_hr['hit_ratio'], 
                color='blue', s=150, edgecolor='black', marker='*', 
                label='Best Speed-Hit Ratio Point')
    ax2.text(best_point_hr['generation_time'], best_point_hr['hit_ratio'], 
             '  Best Combined Point', color='blue', ha='left')
    ax2.legend()
    ax2.grid(True)

    print("\nGenerating visualization charts. Please check the pop-up windows. The program will end after you close the chart windows.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 TeaCache 的 JSON 输出文件，并找出最优参数。")
    parser.add_argument("json_file", default="teacache_analysis.json", nargs='?', type=Path, help="指向 teacache_analysis.json 文件的路径。")
    parser.add_argument(
        "--max_lpips", 
        type=float, 
        default=0.56, 
        help="设置可接受的最大LPIPS距离阈值，用于过滤低质量数据。例如: --max_lpips 0.6"
    )
    args = parser.parse_args()

    json_path = args.json_file
    if not json_path.is_file():
        print(f"错误: 文件不存在 -> {json_path}")
    else:
        print(f"正在读取文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        df, best_results = analyze_runs(analysis_data, args.max_lpips)
        print_results(best_results)
        create_plots(df, best_results, args.max_lpips)
