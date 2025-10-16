package com.sparrowrecsys.online.datamanager;

import com.sparrowrecsys.online.util.PerformanceMonitor;
import com.sparrowrecsys.online.util.Utility;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.Pipeline;
import redis.clients.jedis.Response;

import java.util.*;

/**
 * 批量数据加载优化工具类
 */
public class BatchDataLoader {

    /**
     * 批量加载电影embeddings (优化版)
     */
    public static int batchLoadMovieEmbeddings(Map<Integer, Movie> movieMap, String embKeyPrefix) {
        PerformanceMonitor.startTimer("BatchLoadMovieEmbeddings");
        
        int validEmbCount = 0;
        try (Jedis jedis = OptimizedRedisClient.getInstance()) {
            // 获取所有embedding keys
            Set<String> movieEmbKeys = jedis.keys(embKeyPrefix + "*");
            System.out.println("🔍 Found " + movieEmbKeys.size() + " movie embedding keys in Redis");

            if (movieEmbKeys.isEmpty()) {
                return 0;
            }

            // 使用Pipeline批量查询
            Pipeline pipeline = jedis.pipelined();
            Map<String, Response<String>> responses = new HashMap<>();
            
            for (String movieEmbKey : movieEmbKeys) {
                responses.put(movieEmbKey, pipeline.get(movieEmbKey));
            }
            
            // 批量执行所有查询
            pipeline.sync();
            
            // 处理结果
            for (Map.Entry<String, Response<String>> entry : responses.entrySet()) {
                String movieEmbKey = entry.getKey();
                Response<String> response = entry.getValue();
                
                try {
                    String movieIdStr = movieEmbKey.split(":")[1];
                    int movieId = Integer.parseInt(movieIdStr);
                    Movie movie = movieMap.get(movieId);
                    
                    if (movie != null && response.get() != null) {
                        movie.setEmb(Utility.parseEmbStr(response.get()));
                        validEmbCount++;
                    }
                } catch (Exception e) {
                    System.err.println("⚠️ Error processing movie embedding key: " + movieEmbKey + " - " + e.getMessage());
                }
            }
            
        } catch (Exception e) {
            System.err.println("❌ Error in batch loading movie embeddings: " + e.getMessage());
        }
        
        PerformanceMonitor.endTimer("BatchLoadMovieEmbeddings");
        System.out.println("✅ Batch loading completed. " + validEmbCount + " movie embeddings loaded.");
        
        return validEmbCount;
    }

    /**
     * 批量加载用户embeddings (优化版)
     */
    public static int batchLoadUserEmbeddings(Map<Integer, User> userMap, String embKeyPrefix) {
        PerformanceMonitor.startTimer("BatchLoadUserEmbeddings");
        
        int validEmbCount = 0;
        try (Jedis jedis = OptimizedRedisClient.getInstance()) {
            // 获取所有user embedding keys
            Set<String> userEmbKeys = jedis.keys(embKeyPrefix + "*");
            System.out.println("🔍 Found " + userEmbKeys.size() + " user embedding keys in Redis");

            if (userEmbKeys.isEmpty()) {
                return 0;
            }

            // 使用Pipeline批量查询
            Pipeline pipeline = jedis.pipelined();
            Map<String, Response<String>> responses = new HashMap<>();
            
            for (String userEmbKey : userEmbKeys) {
                responses.put(userEmbKey, pipeline.get(userEmbKey));
            }
            
            // 批量执行所有查询
            pipeline.sync();
            
            // 处理结果
            for (Map.Entry<String, Response<String>> entry : responses.entrySet()) {
                String userEmbKey = entry.getKey();
                Response<String> response = entry.getValue();
                
                try {
                    String userIdStr = userEmbKey.split(":")[1];
                    int userId = Integer.parseInt(userIdStr);
                    User user = userMap.get(userId);
                    
                    if (user != null && response.get() != null) {
                        user.setEmb(Utility.parseEmbStr(response.get()));
                        validEmbCount++;
                    }
                } catch (Exception e) {
                    System.err.println("⚠️ Error processing user embedding key: " + userEmbKey + " - " + e.getMessage());
                }
            }
            
        } catch (Exception e) {
            System.err.println("❌ Error in batch loading user embeddings: " + e.getMessage());
        }
        
        PerformanceMonitor.endTimer("BatchLoadUserEmbeddings");
        System.out.println("✅ Batch loading completed. " + validEmbCount + " user embeddings loaded.");
        
        return validEmbCount;
    }

    /**
     * 批量加载电影特征
     */
    public static int batchLoadMovieFeatures(Map<Integer, Movie> movieMap, String featureKeyPrefix) {
        PerformanceMonitor.startTimer("BatchLoadMovieFeatures");
        
        int validFeatureCount = 0;
        try (Jedis jedis = OptimizedRedisClient.getInstance()) {
            Set<String> featureKeys = jedis.keys(featureKeyPrefix + "*");
            System.out.println("🔍 Found " + featureKeys.size() + " movie feature keys in Redis");

            for (String featureKey : featureKeys) {
                try {
                    String movieIdStr = featureKey.split(":")[1];
                    int movieId = Integer.parseInt(movieIdStr);
                    Movie movie = movieMap.get(movieId);
                    
                    if (movie != null) {
                        Map<String, String> features = jedis.hgetAll(featureKey);
                        if (!features.isEmpty()) {
                            movie.setMovieFeatures(features);
                            validFeatureCount++;
                        }
                    }
                } catch (Exception e) {
                    System.err.println("⚠️ Error processing movie feature key: " + featureKey + " - " + e.getMessage());
                }
            }
            
        } catch (Exception e) {
            System.err.println("❌ Error in batch loading movie features: " + e.getMessage());
        }
        
        PerformanceMonitor.endTimer("BatchLoadMovieFeatures");
        System.out.println("✅ Batch loading completed. " + validFeatureCount + " movie features loaded.");
        
        return validFeatureCount;
    }
}