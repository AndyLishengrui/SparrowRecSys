#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens Embedding Training Script - Python版本
使用PySpark和Gensim训练Item2Vec和User Embedding
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import time
from typing import List, Dict, Tuple
import argparse

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, desc, count, avg
from pyspark.sql.types import StringType, StructType, StructField, FloatType, IntegerType

# Gensim for Word2Vec
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# 进度回调类
class EpochProgressLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print(f"Epoch {self.epoch + 1} started")
        
    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch + 1} completed")
        self.epoch += 1

class MovieLensEmbeddingTrainer:
    def __init__(self, spark_session=None):
        self.spark = spark_session or self._create_spark_session()
        self.ratings_df = None
        self.movies_df = None
        self.user_sequences = []
        
    def _create_spark_session(self):
        """创建Spark会话"""
        return SparkSession.builder \
            .appName("MovieLens Embedding Training") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
    
    def load_data(self, ratings_path: str, movies_path: str = None):
        """加载MovieLens数据"""
        print(f"Loading ratings from: {ratings_path}")
        
        # 读取评分数据
        self.ratings_df = self.spark.read.csv(
            ratings_path, 
            header=True, 
            inferSchema=True
        )
        
        print(f"Total ratings loaded: {self.ratings_df.count()}")
        
        # 如果提供了电影数据路径，也加载电影信息
        if movies_path and os.path.exists(movies_path):
            print(f"Loading movies from: {movies_path}")
            self.movies_df = self.spark.read.csv(
                movies_path,
                header=True,
                inferSchema=True
            )
            print(f"Total movies loaded: {self.movies_df.count()}")
        
        # 显示数据统计
        self._show_data_stats()
    
    def _show_data_stats(self):
        """显示数据统计信息"""
        print("\n=== Data Statistics ===")
        
        # 评分统计
        total_ratings = self.ratings_df.count()
        unique_users = self.ratings_df.select("userId").distinct().count()
        unique_movies = self.ratings_df.select("movieId").distinct().count()
        
        print(f"Total ratings: {total_ratings:,}")
        print(f"Unique users: {unique_users:,}")
        print(f"Unique movies: {unique_movies:,}")
        
        # 评分分布
        print("\nRating distribution:")
        self.ratings_df.groupBy("rating").count().orderBy("rating").show()
        
        # 平均评分
        avg_rating = self.ratings_df.agg(avg("rating")).collect()[0][0]
        print(f"Average rating: {avg_rating:.2f}")
        
        # 最活跃用户
        print("\nTop 10 most active users:")
        self.ratings_df.groupBy("userId") \
            .agg(count("movieId").alias("num_ratings")) \
            .orderBy(desc("num_ratings")) \
            .limit(10).show()
        
        # 最热门电影
        print("\nTop 10 most rated movies:")
        self.ratings_df.groupBy("movieId") \
            .agg(count("userId").alias("num_ratings")) \
            .orderBy(desc("num_ratings")) \
            .limit(10).show()
    
    def generate_user_sequences(self, min_rating: float = 3.5, min_sequence_length: int = 5):
        """生成用户观影序列"""
        print(f"\nGenerating user sequences (min_rating: {min_rating}, min_length: {min_sequence_length})")
        
        # 过滤高评分电影，按时间戳排序
        filtered_ratings = self.ratings_df.filter(col("rating") >= min_rating)
        
        # 按用户分组，收集电影ID序列（按时间戳排序）
        user_sequences_df = filtered_ratings \
            .orderBy("userId", "timestamp") \
            .groupBy("userId") \
            .agg(collect_list("movieId").alias("movie_sequence"))
        
        # 转换为Python列表，过滤短序列
        sequences_data = user_sequences_df.collect()
        
        self.user_sequences = []
        for row in sequences_data:
            sequence = [str(movie_id) for movie_id in row.movie_sequence]
            if len(sequence) >= min_sequence_length:
                self.user_sequences.append(sequence)
        
        print(f"Generated {len(self.user_sequences)} user sequences")
        print(f"Average sequence length: {np.mean([len(seq) for seq in self.user_sequences]):.2f}")
        print(f"Max sequence length: {max([len(seq) for seq in self.user_sequences])}")
        print(f"Min sequence length: {min([len(seq) for seq in self.user_sequences])}")
        
        # 显示示例序列
        print("\nExample sequences:")
        for i, seq in enumerate(self.user_sequences[:3]):
            print(f"User {i+1}: {' -> '.join(seq[:10])}{'...' if len(seq) > 10 else ''}")
    
    def train_item2vec(self, vector_size: int = 100, window: int = 5, min_count: int = 5, 
                      epochs: int = 10, workers: int = 4, output_file: str = None):
        """训练Item2Vec模型"""
        print(f"\n=== Training Item2Vec Model ===")
        print(f"Vector size: {vector_size}")
        print(f"Window size: {window}")
        print(f"Min count: {min_count}")
        print(f"Epochs: {epochs}")
        print(f"Workers: {workers}")
        
        if not self.user_sequences:
            raise ValueError("No user sequences available. Please run generate_user_sequences() first.")
        
        # 创建进度回调
        progress_logger = EpochProgressLogger()
        
        # 训练Word2Vec模型
        start_time = time.time()
        
        model = Word2Vec(
            sentences=self.user_sequences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            sg=1,  # Skip-gram
            negative=5,  # 负采样
            callbacks=[progress_logger]
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # 模型统计
        vocab_size = len(model.wv.key_to_index)
        print(f"Vocabulary size: {vocab_size}")
        
        # 保存模型
        if output_file:
            print(f"Saving model to: {output_file}")
            model.save(output_file)
            
            # 保存向量到CSV文件
            csv_file = output_file.replace('.model', '_vectors.csv')
            self._save_vectors_to_csv(model, csv_file)
        
        # 测试相似性
        self._test_similarity(model)
        
        return model
    
    def _save_vectors_to_csv(self, model, csv_file: str):
        """保存向量到CSV文件"""
        print(f"Saving vectors to CSV: {csv_file}")
        
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("movieId,vector\n")
            for movie_id in model.wv.key_to_index:
                vector_str = ' '.join([str(x) for x in model.wv[movie_id]])
                f.write(f"{movie_id},{vector_str}\n")
        
        print(f"Vectors saved to {csv_file}")
    
    def _test_similarity(self, model, test_movie_ids: List[str] = None):
        """测试电影相似性"""
        print("\n=== Testing Movie Similarity ===")
        
        if test_movie_ids is None:
            # 选择一些常见的电影ID进行测试
            vocab_keys = list(model.wv.key_to_index.keys())
            test_movie_ids = vocab_keys[:5] if len(vocab_keys) >= 5 else vocab_keys
        
        for movie_id in test_movie_ids:
            if movie_id in model.wv.key_to_index:
                try:
                    similar_movies = model.wv.most_similar(movie_id, topn=5)
                    print(f"\nMovies similar to {movie_id}:")
                    for similar_movie, similarity in similar_movies:
                        print(f"  {similar_movie}: {similarity:.4f}")
                except Exception as e:
                    print(f"Error finding similar movies for {movie_id}: {e}")
            else:
                print(f"Movie {movie_id} not in vocabulary")
    
    def generate_user_embeddings(self, item2vec_model, output_file: str = None):
        """基于Item2Vec生成用户向量"""
        print("\n=== Generating User Embeddings ===")
        
        if not self.user_sequences:
            raise ValueError("No user sequences available.")
        
        user_embeddings = {}
        vector_size = item2vec_model.vector_size
        
        for i, sequence in enumerate(self.user_sequences):
            user_id = f"user_{i}"
            user_vector = np.zeros(vector_size)
            valid_movies = 0
            
            # 计算用户观看电影的平均向量
            for movie_id in sequence:
                if movie_id in item2vec_model.wv.key_to_index:
                    user_vector += item2vec_model.wv[movie_id]
                    valid_movies += 1
            
            if valid_movies > 0:
                user_vector /= valid_movies
                user_embeddings[user_id] = user_vector
        
        print(f"Generated embeddings for {len(user_embeddings)} users")
        
        # 保存用户向量
        if output_file:
            print(f"Saving user embeddings to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("userId,vector\n")
                for user_id, vector in user_embeddings.items():
                    vector_str = ' '.join([str(x) for x in vector])
                    f.write(f"{user_id},{vector_str}\n")
        
        return user_embeddings
    
    def train_graph_embedding(self, walk_length: int = 10, num_walks: int = 80, 
                            vector_size: int = 100, window: int = 5, epochs: int = 10,
                            output_file: str = None):
        """训练图嵌入（基于随机游走）"""
        print("\n=== Training Graph Embedding ===")
        print(f"Walk length: {walk_length}")
        print(f"Number of walks per node: {num_walks}")
        
        # 构建转移矩阵
        transition_matrix = self._build_transition_matrix()
        
        # 生成随机游走序列
        walk_sequences = self._generate_random_walks(
            transition_matrix, walk_length, num_walks
        )
        
        print(f"Generated {len(walk_sequences)} random walk sequences")
        
        # 训练Word2Vec模型
        model = Word2Vec(
            sentences=walk_sequences,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            sg=1,
            negative=5,
            workers=4
        )
        
        print(f"Graph embedding vocabulary size: {len(model.wv.key_to_index)}")
        
        # 保存模型
        if output_file:
            model.save(output_file)
            csv_file = output_file.replace('.model', '_vectors.csv')
            self._save_vectors_to_csv(model, csv_file)
        
        return model
    
    def _build_transition_matrix(self):
        """构建电影转移矩阵"""
        print("Building transition matrix...")
        
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        # 统计电影之间的转移次数
        for sequence in self.user_sequences:
            for i in range(len(sequence) - 1):
                from_movie = sequence[i]
                to_movie = sequence[i + 1]
                transition_counts[from_movie][to_movie] += 1
        
        # 转换为概率
        transition_matrix = {}
        for from_movie, to_movies in transition_counts.items():
            total_transitions = sum(to_movies.values())
            transition_matrix[from_movie] = {
                to_movie: count / total_transitions 
                for to_movie, count in to_movies.items()
            }
        
        print(f"Transition matrix built for {len(transition_matrix)} movies")
        return transition_matrix
    
    def _generate_random_walks(self, transition_matrix: Dict, walk_length: int, num_walks: int):
        """生成随机游走序列"""
        print("Generating random walks...")
        
        all_movies = list(transition_matrix.keys())
        walk_sequences = []
        
        for _ in range(num_walks):
            for start_movie in all_movies:
                walk = self._single_random_walk(transition_matrix, start_movie, walk_length)
                if len(walk) > 1:
                    walk_sequences.append(walk)
        
        return walk_sequences
    
    def _single_random_walk(self, transition_matrix: Dict, start_movie: str, walk_length: int):
        """单次随机游走"""
        walk = [start_movie]
        current_movie = start_movie
        
        for _ in range(walk_length - 1):
            if current_movie not in transition_matrix:
                break
            
            next_movies = list(transition_matrix[current_movie].keys())
            probabilities = list(transition_matrix[current_movie].values())
            
            # 按概率选择下一个电影
            next_movie = np.random.choice(next_movies, p=probabilities)
            walk.append(next_movie)
            current_movie = next_movie
        
        return walk
    
    def close(self):
        """关闭Spark会话"""
        if self.spark:
            self.spark.stop()

def main():
    parser = argparse.ArgumentParser(description='MovieLens Embedding Training')
    parser.add_argument('--ratings_path', required=True, help='Path to ratings.csv')
    parser.add_argument('--movies_path', help='Path to movies.csv (optional)')
    parser.add_argument('--output_dir', default='./embeddings', help='Output directory')
    parser.add_argument('--vector_size', type=int, default=100, help='Embedding vector size')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--min_rating', type=float, default=3.5, help='Minimum rating threshold')
    parser.add_argument('--min_sequence_length', type=int, default=5, help='Minimum sequence length')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化训练器
    trainer = MovieLensEmbeddingTrainer()
    
    try:
        # 加载数据
        trainer.load_data(args.ratings_path, args.movies_path)
        
        # 生成用户序列
        trainer.generate_user_sequences(
            min_rating=args.min_rating,
            min_sequence_length=args.min_sequence_length
        )
        
        # 训练Item2Vec
        item2vec_model = trainer.train_item2vec(
            vector_size=args.vector_size,
            epochs=args.epochs,
            output_file=os.path.join(args.output_dir, 'item2vec.model')
        )
        
        # 生成用户向量
        trainer.generate_user_embeddings(
            item2vec_model,
            output_file=os.path.join(args.output_dir, 'user_embeddings.csv')
        )
        
        # 训练图嵌入
        graph_model = trainer.train_graph_embedding(
            vector_size=args.vector_size,
            epochs=args.epochs,
            output_file=os.path.join(args.output_dir, 'graph_embedding.model')
        )
        
        print(f"\n=== Training Completed ===")
        print(f"Models saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()

if __name__ == "__main__":
    main()
