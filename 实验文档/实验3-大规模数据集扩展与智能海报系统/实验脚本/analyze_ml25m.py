#!/usr/bin/env python3
"""
MovieLens 25M数据集分析工具
用于分析大规模数据集的质量和分布特征
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def analyze_dataset():
    """
    分析MovieLens 25M数据集
    """
    print("=== MovieLens 25M 数据集分析 ===")
    
    # 检查文件存在性
    if not os.path.exists('ml-25m/ratings.csv'):
        print("错误: 找不到MovieLens 25M数据文件")
        print("请先下载并解压ml-25m.zip")
        return None
    
    # 读取数据
    print("读取数据文件...")
    ratings = pd.read_csv('ml-25m/ratings.csv')
    movies = pd.read_csv('ml-25m/movies.csv')
    links = pd.read_csv('ml-25m/links.csv')
    
    print(f"评分数据: {len(ratings):,} 条记录")
    print(f"电影数据: {len(movies):,} 部电影")
    print(f"用户数量: {ratings['userId'].nunique():,} 个用户")
    print(f"链接数据: {len(links):,} 条外部链接")
    
    # 评分分布分析
    print("\\n=== 评分统计分析 ===")
    print("评分分布:")
    print(ratings['rating'].value_counts().sort_index())
    print("\\n评分统计:")
    print(ratings['rating'].describe())
    
    # 电影评分次数分布
    print("\\n=== 电影评分次数分析 ===")
    movie_counts = ratings['movieId'].value_counts()
    print(f"平均每部电影评分次数: {movie_counts.mean():.1f}")
    print(f"评分次数中位数: {movie_counts.median():.1f}")
    print(f"评分最多的电影: {movie_counts.iloc[0]} 次")
    print(f"评分最少的电影: {movie_counts.iloc[-1]} 次")
    
    # 用户评分次数分布
    print("\\n=== 用户评分次数分析 ===")
    user_counts = ratings['userId'].value_counts()
    print(f"平均每用户评分次数: {user_counts.mean():.1f}")
    print(f"用户评分次数中位数: {user_counts.median():.1f}")
    print(f"最活跃用户评分: {user_counts.iloc[0]} 次")
    print(f"最少评分用户: {user_counts.iloc[-1]} 次")
    
    # 筛选条件分析
    print("\\n=== 数据筛选建议 ===")
    for min_ratings in [50, 100, 200, 500]:
        qualified_movies = movie_counts[movie_counts >= min_ratings]
        qualified_ratings = ratings[ratings['movieId'].isin(qualified_movies.index)]
        print(f"至少{min_ratings}次评分的电影: {len(qualified_movies):,} 部, "
              f"占总评分: {len(qualified_ratings)/len(ratings)*100:.1f}%")
    
    # 类型分析
    print("\\n=== 电影类型分析 ===")
    genre_counts = {}
    for _, row in movies.iterrows():
        genres = str(row['genres']).split('|')
        for genre in genres:
            genre = genre.strip()
            if genre and genre != '(no genres listed)':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    print("电影类型分布 (前10):")
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    for genre, count in sorted_genres[:10]:
        print(f"  {genre}: {count:,} 部电影")
    
    # 时间分析
    print("\\n=== 评分时间分析 ===")
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    print(f"评分时间范围: {ratings['datetime'].min()} 到 {ratings['datetime'].max()}")
    
    # 年度评分统计
    ratings['year'] = ratings['datetime'].dt.year
    yearly_ratings = ratings['year'].value_counts().sort_index()
    print(f"\\n年度评分统计 (最近5年):")
    for year in sorted(yearly_ratings.index)[-5:]:
        print(f"  {year}: {yearly_ratings[year]:,} 条评分")
    
    # 推荐筛选参数
    print("\\n=== 推荐筛选参数 ===")
    min_ratings = 100
    qualified_movies = movie_counts[movie_counts >= min_ratings]
    
    print(f"建议筛选条件: 每部电影至少 {min_ratings} 次评分")
    print(f"筛选后电影数量: {len(qualified_movies):,} 部")
    print(f"数据质量评估: 高质量电影占比 {len(qualified_movies)/len(movies)*100:.1f}%")
    
    return qualified_movies.index.tolist()

def generate_visualization(save_plots=False):
    """
    生成数据可视化图表
    """
    try:
        print("\\n=== 生成数据可视化 ===")
        ratings = pd.read_csv('ml-25m/ratings.csv')
        movies = pd.read_csv('ml-25m/movies.csv')
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 评分分布
        ratings['rating'].hist(bins=10, ax=axes[0,0], edgecolor='black')
        axes[0,0].set_title('评分分布')
        axes[0,0].set_xlabel('评分')
        axes[0,0].set_ylabel('频次')
        
        # 2. 电影评分次数分布
        movie_counts = ratings['movieId'].value_counts()
        movie_counts.hist(bins=50, ax=axes[0,1], edgecolor='black')
        axes[0,1].set_title('电影评分次数分布')
        axes[0,1].set_xlabel('评分次数')
        axes[0,1].set_ylabel('电影数量')
        axes[0,1].set_xscale('log')
        
        # 3. 用户评分次数分布
        user_counts = ratings['userId'].value_counts()
        user_counts.hist(bins=50, ax=axes[1,0], edgecolor='black')
        axes[1,0].set_title('用户评分次数分布')
        axes[1,0].set_xlabel('评分次数')
        axes[1,0].set_ylabel('用户数量')
        axes[1,0].set_xscale('log')
        
        # 4. 时间分布
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year'] = ratings['datetime'].dt.year
        yearly_counts = ratings['year'].value_counts().sort_index()
        yearly_counts.plot(kind='line', ax=axes[1,1])
        axes[1,1].set_title('年度评分数量趋势')
        axes[1,1].set_xlabel('年份')
        axes[1,1].set_ylabel('评分数量')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ml25m_analysis.png', dpi=300, bbox_inches='tight')
            print("可视化图表已保存: ml25m_analysis.png")
        
        plt.show()
        
    except ImportError:
        print("跳过可视化: matplotlib或seaborn未安装")
    except Exception as e:
        print(f"可视化生成失败: {e}")

def generate_report(qualified_movies):
    """
    生成分析报告
    """
    report_content = f"""
MovieLens 25M 数据集分析报告
===========================
生成时间: {datetime.now()}

数据概览:
- 评分记录: 25,000,095 条
- 电影数量: 62,424 部
- 用户数量: 162,541 个
- 时间跨度: 1995-2019年

数据质量:
- 推荐筛选: 至少100次评分的电影
- 筛选后电影: {len(qualified_movies) if qualified_movies else 0:,} 部
- 数据质量: 高质量电影比例高

推荐用途:
- 大规模推荐系统训练
- 协同过滤算法验证
- 深度学习模型实验
- A/B测试数据源

下一步操作:
1. 运行数据转换脚本
2. 训练embedding模型
3. 集成到SparrowRecSys系统
"""
    
    with open('ml25m_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("\\n分析报告已保存: ml25m_analysis_report.txt")

def main():
    """
    主函数
    """
    print("MovieLens 25M 数据集分析工具")
    print("=" * 50)
    
    try:
        # 执行分析
        qualified_movies = analyze_dataset()
        
        if qualified_movies:
            # 生成可视化（可选）
            try:
                generate_visualization(save_plots=True)
            except:
                print("跳过可视化生成")
            
            # 生成报告
            generate_report(qualified_movies)
            
            print(f"\\n=== 分析完成 ===")
            print(f"推荐使用 {len(qualified_movies):,} 部高质量电影进行训练")
            print("下一步: 运行 convert_ml25m_to_sparrow.py 进行数据转换")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()