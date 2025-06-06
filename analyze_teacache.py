try:
  import json
  import argparse
  import pandas as pd
  import matplotlib.pyplot as plt
  from pathlib import Path
  
  def analyze_runs(data):
      if not data:
          print("错误：JSON文件中没有数据。")
          return None, None
  
      # 预处理
      df = pd.DataFrame(data)
  
      # 过滤掉没有 'coefficients' 的无效运行记录
      df = df.dropna(subset=['coefficients'])
      if df.empty:
          print("错误：未找到包含'coefficients'的有效运行记录。")
          return None, None
  
      # 计算缓存命中率
      # 为了避免除以0的错误，当 total_inferences 为0时，命中率也为0
      df['hit_ratio'] = (df['cache_hits'] / df['total_inferences']).fillna(0)
  
      # 2. 寻找基准和计算节省时间
      # 基准运行指的是不开启缓存的运行 (rel_l1_thresh == 0)
      baseline_runs = df[df['rel_l1_thresh'] == 0]
      if baseline_runs.empty:
          print("警告：未找到基准运行 (rel_l1_thresh == 0)，无法计算'节省时间'和'综合得分'。")
          baseline_time = df['generation_time'].max() # 如果没有基准，使用最慢时间作为估算
      else:
          baseline_time = baseline_runs['generation_time'].min()
      
      print(f"基准运行时间 (最快无缓存): {baseline_time:.2f} 秒")
      
      # 计算每条记录节省的时间
      df['time_saved'] = baseline_time - df['generation_time']
      # 过滤掉比基准还慢的“负节省”情况
      df.loc[df['time_saved'] < 0, 'time_saved'] = 0
  
      # 3. 计算两种综合效率得分
      # 得分越高越好
      
      # LPIPS得分 = 节省的时间 / LPIPS距离 (LPIPS越小越好)
      if 'lpips_distance' in df.columns:
          # 避免除以0或空值
          df['score_lpips'] = (df['time_saved'] / df['lpips_distance'].replace(0, float('inf'))).fillna(0)
      else:
          print("警告: 数据中不包含 'lpips_distance'，跳过LPIPS相关分析。")
          df['score_lpips'] = 0
  
      # 命中率得分 = 节省的时间 * 缓存命中率 (两者都越大越好)
      df['score_hit_ratio'] = df['time_saved'] * df['hit_ratio']
  
      # 4. 找出各项最佳参数
      results = {}
      
      # 仅在有缓存的运行中寻找最优值
      cached_runs = df[df['rel_l1_thresh'] > 0]
      if cached_runs.empty:
          print("警告: 未找到任何开启缓存的运行记录，无法推荐最优参数。")
          return df, None
  
      # 找到各项指标最好的那一行记录的索引
      best_indices = {
          "最快生成速度": cached_runs['generation_time'].idxmin(),
          "最高缓存命中率": cached_runs['hit_ratio'].idxmax(),
          "最佳综合效率 (命中率)": cached_runs['score_hit_ratio'].idxmax(),
      }
      if df['score_lpips'].sum() > 0:
          best_indices["最低LPIPS (最佳画质)"] = cached_runs['lpips_distance'].idxmin()
          best_indices["最佳综合效率 (LPIPS)"] = cached_runs['score_lpips'].idxmax()
  
      # 整理结果
      for name, idx in best_indices.items():
          # .loc[idx] 可以获取索引对应的整行数据
          best_run = df.loc[idx]
          results[name] = {
              "coefficients": best_run['coefficients'],
              "value": {
                  "生成时间": f"{best_run['generation_time']:.2f}s",
                  "命中率": f"{best_run['hit_ratio']:.2%}",
                  "LPIPS": f"{best_run.get('lpips_distance', 'N/A'):.4f}" if pd.notna(best_run.get('lpips_distance')) else "N/A",
                  "综合得分(命中率)": f"{best_run['score_hit_ratio']:.2f}",
                  "综合得分(LPIPS)": f"{best_run['score_lpips']:.2f}",
              }
          }
          
      return df, results
  
  def print_results(results):
      """
      格式化打印分析结果。
      """
      if not results:
          return
          
      print("\n" + "="*25 + " 分析结果 " + "="*25)
      for name, data in results.items():
          print(f"\n--- {name}")
          print(f"  最佳Coefficients: {data['coefficients']}")
          print("     相关指标:")
          for key, val in data['value'].items():
              print(f"       - {key}: {val}")
      print("\n" + "="*62)
  
  
  def create_plots(df, results):
      """
      使用matplotlib创建数据可视化图表。
      """
      if df is None or results is None:
          print("数据不足，无法生成图表。")
          return
  
      # 过滤掉无缓存的基准点，让图表更清晰
      plot_df = df[df['rel_l1_thresh'] > 0].copy()
      if plot_df.empty:
          print("无缓存运行数据，无法生成图表。")
          return
  
      # 图表1: 时间 vs LPIPS (速度与质量的权衡)
      if 'score_lpips' in plot_df.columns and plot_df['lpips_distance'].notna().any():
          plt.style.use('seaborn-v0_8-darkgrid')
          fig1, ax1 = plt.subplots(figsize=(12, 8))
          
          scatter1 = ax1.scatter(
              plot_df['generation_time'], 
              plot_df['lpips_distance'], 
              c=plot_df['score_lpips'], 
              cmap='viridis', 
              alpha=0.7,
              s=50 # s是点的大小
          )
          fig1.colorbar(scatter1, label='综合效率得分 (LPIPS Score)')
          
          ax1.set_title('生成速度 vs 图像质量 (LPIPS)', fontsize=16)
          ax1.set_xlabel('生成时间 (秒) - 越低越好', fontsize=12)
          ax1.set_ylabel('LPIPS 距离 - 越低越好', fontsize=12)
          
          # 在图上标注出最佳的点
          best_lpips_score_idx = plot_df['score_lpips'].idxmax()
          best_point = plot_df.loc[best_lpips_score_idx]
          ax1.scatter(best_point['generation_time'], best_point['lpips_distance'], color='red', s=150, ec='black', marker='*', label='最佳LPIPS综合效率')
          ax1.text(best_point['generation_time'], best_point['lpips_distance'], '  最佳综合点', color='red', ha='left')
  
          ax1.legend()
          ax1.grid(True)
  
      # 图表2: 时间 vs 缓存命中率 (速度与效率的权衡)
      fig2, ax2 = plt.subplots(figsize=(12, 8))
  
      scatter2 = ax2.scatter(
          plot_df['generation_time'], 
          plot_df['hit_ratio'], 
          c=plot_df['score_hit_ratio'], 
          cmap='plasma', 
          alpha=0.7,
          s=50
      )
      fig2.colorbar(scatter2, label='综合效率得分 (Hit Ratio Score)')
  
      ax2.set_title('生成速度 vs 缓存命中率', fontsize=16)
      ax2.set_xlabel('生成时间 (秒) - 越低越好', fontsize=12)
      ax2.set_ylabel('缓存命中率 - 越高越好', fontsize=12)
      ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # Y轴显示百分比
  
      # 标注最佳点
      best_hr_score_idx = plot_df['score_hit_ratio'].idxmax()
      best_point_hr = plot_df.loc[best_hr_score_idx]
      ax2.scatter(best_point_hr['generation_time'], best_point_hr['hit_ratio'], color='blue', s=150, ec='black', marker='*', label='最佳命中率综合效率')
      ax2.text(best_point_hr['generation_time'], best_point_hr['hit_ratio'], '  最佳综合点', color='blue', ha='left')
  
      ax2.legend()
      ax2.grid(True)
  
      print("\n正在生成可视化图表，请在弹出的窗口中查看。关闭图表窗口后程序将结束。")
      plt.tight_layout()
      plt.show()
  
  
  if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="分析 TeaCache 的 JSON 输出文件，并找出最优参数。")
      parser.add_argument("json_file", type=Path, help="指向 teacache_analysis.json 文件的路径。")
      args = parser.parse_args()
  
      json_path = args.json_file
      if not json_path.is_file():
          print(f"错误: 文件不存在 -> {json_path}")
      else:
          print(f"正在读取文件: {json_path}")
          with open(json_path, 'r', encoding='utf-8') as f:
              analysis_data = json.load(f)
  
          df, best_results = analyze_runs(analysis_data)
          print_results(best_results)
          create_plots(df, best_results)
except:
  pass
