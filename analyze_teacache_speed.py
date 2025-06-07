import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_fastest_runs(data, max_lpips_thresh=None, top_n=5):
    """
    分析数据并找出LPIPS值小于阈值的最快的n组运行数据。
    """
    if not data:
        print("错误：JSON文件中没有数据。")
        return None, None

    # 数据预处理
    df = pd.DataFrame(data)
    df = df.dropna(subset=['coefficients', 'generation_time'])
    if df.empty:
        print("错误：未找到包含'coefficients'和'generation_time'的有效运行记录。")
        return None, None

    # 检查并应用LPIPS过滤
    has_lpips = 'lpips_distance' in df.columns
    filtered_df = df.copy()
    
    if has_lpips and max_lpips_thresh is not None and max_lpips_thresh > 0:
        print("\n" + "-"*15 + f" 应用质量门槛 (LPIPS <= {max_lpips_thresh}) " + "-"*15)
        original_count = len(filtered_df)
        filtered_df = filtered_df.dropna(subset=['lpips_distance'])
        filtered_df = filtered_df[filtered_df['lpips_distance'] <= max_lpips_thresh]
        filtered_count = len(filtered_df)
        print(f"已从 {original_count} 条记录中筛选出 {filtered_count} 条符合条件的记录。")
        
        if filtered_df.empty:
            print("警告：应用阈值后，没有符合条件的运行记录。")
            return None, None
    elif max_lpips_thresh is not None and max_lpips_thresh > 0:
        print("警告: 数据中不包含 'lpips_distance'，无法应用质量门槛。")

    # 按生成时间排序并选择最快的n组
    sorted_df = filtered_df.sort_values(by='generation_time')
    fastest_n = sorted_df.head(top_n)
    
    if fastest_n.empty:
        print("警告：没有找到符合条件的运行记录。")
        return None, None
    
    # 准备结果数据
    results = {
        f"第 {i+1} 快": {
            "coefficients": row['coefficients'],
            "value": {
                "生成时间": f"{row['generation_time']:.2f}s",
                "LPIPS": f"{row.get('lpips_distance', 'N/A'):.4f}" if pd.notna(row.get('lpips_distance')) else "N/A",
                "rel_l1_thresh": row.get('rel_l1_thresh', 'N/A'),
                "rel_l2_thresh": row.get('rel_l2_thresh', 'N/A'),
            }
        }
        for i, (_, row) in enumerate(fastest_n.iterrows())
    }
    
    return filtered_df, results

def print_fastest_results(results, max_lpips=None):
    if not results:
        return
        
    print("\n" + "="*25 + " 分析结果 " + "="*25)
    if max_lpips is not None:
        print(f"条件: LPIPS 距离 <= {max_lpips}")
    print(f"找到最快的 {len(results)} 组运行参数:\n")
    
    for rank, data in results.items():
        print(f"{rank}:")
        print(f"  最佳Coefficients: {data['coefficients']}")
        print("     相关指标:")
        for key, val in data['value'].items():
            print(f"       - {key}: {val}")
        print()
    print("="*62)

def create_speed_lpips_plot(df, results, max_lpips_thresh=None):
    """创建生成速度与LPIPS值的散点图，标记最快的n组数据"""
    if df is None or results is None or 'lpips_distance' not in df.columns:
        print("无法创建速度-LPIPS图表：数据不足或缺少LPIPS列。")
        return

    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 绘制所有点
    scatter = plt.scatter(df['generation_time'], df['lpips_distance'], 
                         alpha=0.6, s=50, c='lightblue', label='All Runs')
    
    # 提取最快n组的数据点
    fastest_times = [list(r['value'].values())[0] for r in results.values()]
    fastest_times = [float(t[:-1]) for t in fastest_times]  # 移除's'后缀并转换为float
    
    if 'lpips_distance' in df.columns:
        fastest_lpips = [list(r['value'].values())[1] for r in results.values()]
        fastest_lpips = [float(l) if l != 'N/A' else None for l in fastest_lpips]
    else:
        fastest_lpips = [None] * len(results)
    
    # 标记最快的n组
    for i, (time, lpips) in enumerate(zip(fastest_times, fastest_lpips)):
        if lpips is not None:
            plt.scatter(time, lpips, color='red', s=100, edgecolor='black', 
                       marker='*', label=f'Fastest #{i+1}' if i == 0 else "")
            plt.text(time, lpips, f'  #{i+1}', color='red', ha='left', fontweight='bold')
    
    plt.title('Generation Speed vs LPIPS Distance', fontsize=16)
    plt.xlabel('Generation Time (seconds) - Lower is better', fontsize=12)
    plt.ylabel('LPIPS Distance - Lower is better', fontsize=12)
    
    if max_lpips_thresh is not None:
        plt.axhline(y=max_lpips_thresh, color='r', linestyle='--', alpha=0.5, 
                   label=f'LPIPS Threshold = {max_lpips_thresh}')
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析TeaCache的JSON输出文件，找出LPIPS值小于阈值的最快的n组运行参数。")
    parser.add_argument("json_file", default="teacache_analysis.json", nargs='?', type=Path, 
                        help="指向teacache_analysis.json文件的路径。")
    parser.add_argument(
        "--max_lpips", 
        type=float, 
        default=0.455, 
        help="设置可接受的最大LPIPS距离阈值，用于过滤低质量数据。例如: --max_lpips 0.6"
    )
    parser.add_argument(
        "--top_n", 
        type=int, 
        default=5, 
        help="指定要返回的最快运行参数的数量。例如: --top_n 3"
    )
    args = parser.parse_args()

    json_path = args.json_file
    if not json_path.is_file():
        print(f"错误: 文件不存在 -> {json_path}")
    else:
        print(f"正在读取文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        df, best_results = analyze_fastest_runs(analysis_data, args.max_lpips, args.top_n)
        print_fastest_results(best_results, args.max_lpips)
        
        # 如果有LPIPS数据，创建可视化图表
        if df is not None and 'lpips_distance' in df.columns:
            create_speed_lpips_plot(df, best_results, args.max_lpips)
