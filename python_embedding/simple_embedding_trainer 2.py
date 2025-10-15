#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens Embedding Training Script - 简化版本
使用pandas和Gensim训练Item2Vec和User Embedding，无需Spark
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random
import time
from typing import List, Dict, Tuple
import argparse

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

class SimpleEmbeddingTrainer:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.user_sequences = []
        
    def load_data(self, ratings_path: str, movies_path: str = None):
        """加载MovieLens数据"""
        print(f"Loading ratings from: {ratings_path}")
        
        # 读取评分数据
        self.ratings_df = pd.read_csv(ratings_path)
        print(f"Total ratings loaded: {len(self.ratings_df):,}")
        
        # 如果提供了电影数据路径，也加载电影信息
        if movies_path and os.path.exists(movies_path):
            print(f"Loading movies from: {movies_path}")
            self.movies_df = pd.read_csv(movies_path)
            print(f"Total movies loaded: {len(self.movies_df):,}")
        
        # 显示数据统计
        self._show_data_stats()
    
    def _show_data_stats(self):
        """显示数据统计信息"""
        print("\n=== Data Statistics ===")
        
        # 基本统计
        total_ratings = len(self.ratings_df)
        unique_users = self.ratings_df['userId'].nunique()
        unique_movies = self.ratings_df['movieId'].nunique()
        
        print(f"Total ratings: {total_ratings:,}")
        print(f"Unique users: {unique_users:,}")
        print(f"Unique movies: {unique_movies:,}")
        
        # 评分分布
        print("\nRating distribution:")
        rating_dist = self.ratings_df['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"  {rating}: {count:,} ({count/total_ratings*100:.1f}%)")
        
        # 平均评分
        avg_rating = self.ratings_df['rating'].mean()
        print(f"Average rating: {avg_rating:.2f}")
        
        # 最活跃用户
        print("\nTop 10 most active users:")
        top_users = self.ratings_df['userId'].value_counts().head(10)
        for user_id, count in top_users.items():
            print(f"  User {user_id}: {count} ratings")
        
        # 最热门电影
        print("\nTop 10 most rated movies:")
        top_movies = self.ratings_df['movieId'].value_counts().head(10)
        for movie_id, count in top_movies.items():
            movie_title = "Unknown"
            if self.movies_df is not None:
                movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
                if not movie_row.empty:
                    movie_title = movie_row.iloc[0]['title']
            print(f"  Movie {movie_id} ({movie_title}): {count} ratings")
    
    def generate_user_sequences(self, min_rating: float = 3.5, min_sequence_length: int = 5, 
                               max_users: int = None):
        """生成用户观影序列"""
        print(f"\nGenerating user sequences (min_rating: {min_rating}, min_length: {min_sequence_length})")
        
        # 过滤高评分电影
        filtered_ratings = self.ratings_df[self.ratings_df['rating'] >= min_rating].copy()
        print(f"Filtered to {len(filtered_ratings):,} high-rating records")
        
        # 按用户分组，按时间戳排序
        user_groups = filtered_ratings.groupby('userId')
        
        sequences = []
        processed_users = 0
        
        for user_id, user_data in user_groups:
            # 按时间戳排序
            user_data_sorted = user_data.sort_values('timestamp')
            sequence = [str(movie_id) for movie_id in user_data_sorted['movieId'].tolist()]
            
            if len(sequence) >= min_sequence_length:
                sequences.append(sequence)
                processed_users += 1
                
                # 限制用户数量（用于测试）
                if max_users and processed_users >= max_users:
                    break
        
        self.user_sequences = sequences
        
        print(f"Generated {len(self.user_sequences)} user sequences")
        if self.user_sequences:
            seq_lengths = [len(seq) for seq in self.user_sequences]
            print(f"Average sequence length: {np.mean(seq_lengths):.2f}")
            print(f"Max sequence length: {max(seq_lengths)}")
            print(f"Min sequence length: {min(seq_lengths)}")
            
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
        
        vectors_data = []
        for movie_id in model.wv.key_to_index:
            vector_str = ' '.join([str(x) for x in model.wv[movie_id]])
            vectors_data.append({
                'movieId': movie_id,
                'vector': vector_str
            })
        
        vectors_df = pd.DataFrame(vectors_data)
        vectors_df.to_csv(csv_file, index=False)
        print(f"Vectors saved to {csv_file}")
        
        # 也保存兼容原格式的文件
        compat_file = csv_file.replace('_vectors.csv', 'Emb.csv')
        with open(compat_file, 'w', encoding='utf-8') as f:
            for movie_id in model.wv.key_to_index:
                vector_str = ' '.join([str(x) for x in model.wv[movie_id]])
                f.write(f"{movie_id}:{vector_str}\n")
        print(f"Compatible format saved to {compat_file}")
    
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
                    movie_title = self._get_movie_title(movie_id)
                    print(f"\nMovies similar to {movie_id} ({movie_title}):")
                    for similar_movie, similarity in similar_movies:
                        similar_title = self._get_movie_title(similar_movie)
                        print(f"  {similar_movie} ({similar_title}): {similarity:.4f}")
                except Exception as e:
                    print(f"Error finding similar movies for {movie_id}: {e}")
            else:
                print(f"Movie {movie_id} not in vocabulary")
    
    def _get_movie_title(self, movie_id: str) -> str:
        """获取电影标题"""
        if self.movies_df is not None:
            try:
                movie_row = self.movies_df[self.movies_df['movieId'] == int(movie_id)]
                if not movie_row.empty:
                    return movie_row.iloc[0]['title']
            except:
                pass
        return "Unknown"
    
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
            
            # 保存为CSV格式
            user_data = []
            for user_id, vector in user_embeddings.items():
                vector_str = ' '.join([str(x) for x in vector])
                user_data.append({
                    'userId': user_id,
                    'vector': vector_str
                })
            
            user_df = pd.DataFrame(user_data)
            user_df.to_csv(output_file, index=False)
            
            # 也保存兼容原格式的文件
            compat_file = output_file.replace('.csv', 'Emb.csv')
            with open(compat_file, 'w', encoding='utf-8') as f:
                for user_id, vector in user_embeddings.items():
                    vector_str = ' '.join([str(x) for x in vector])
                    f.write(f"{user_id}:{vector_str}\n")
            print(f"Compatible format saved to {compat_file}")
        
        return user_embeddings
    
    def train_graph_embedding(self, walk_length: int = 10, num_walks: int = 20, 
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
        
        if not walk_sequences:
            print("No random walk sequences generated. Skipping graph embedding.")
            return None
        
        # 训练Word2Vec模型
        progress_logger = EpochProgressLogger()
        
        model = Word2Vec(
            sentences=walk_sequences,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            sg=1,
            negative=5,
            workers=4,
            min_count=1,  # 降低最小频次要求
            callbacks=[progress_logger]
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
            if total_transitions > 0:
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
        
        if not all_movies:
            return walk_sequences
        
        for _ in range(num_walks):
            for start_movie in all_movies:
                walk = self._single_random_walk(transition_matrix, start_movie, walk_length)
                if len(walk) > 1:
                    walk_sequences.append(walk)
                    
                # 限制序列数量以避免内存问题
                if len(walk_sequences) > 10000:
                    return walk_sequences
        
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
            
            try:
                # 按概率选择下一个电影
                next_movie = np.random.choice(next_movies, p=probabilities)
                walk.append(next_movie)
                current_movie = next_movie
            except:
                break
        
        return walk

def main():
    parser = argparse.ArgumentParser(description='MovieLens Embedding Training - Simple Version')
    parser.add_argument('--ratings_path', required=True, help='Path to ratings.csv')
    parser.add_argument('--movies_path', help='Path to movies.csv (optional)')
    parser.add_argument('--output_dir', default='./embeddings', help='Output directory')
    parser.add_argument('--vector_size', type=int, default=100, help='Embedding vector size')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--min_rating', type=float, default=3.5, help='Minimum rating threshold')
    parser.add_argument('--min_sequence_length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max_users', type=int, help='Maximum number of users to process (for testing)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化训练器
    trainer = SimpleEmbeddingTrainer()
    
    try:
        # 加载数据
        trainer.load_data(args.ratings_path, args.movies_path)
        
        # 生成用户序列
        trainer.generate_user_sequences(
            min_rating=args.min_rating,
            min_sequence_length=args.min_sequence_length,
            max_users=args.max_users
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
        
        # 列出生成的文件
        print("\nGenerated files:")
        for file in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file}: {file_size:,} bytes")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
