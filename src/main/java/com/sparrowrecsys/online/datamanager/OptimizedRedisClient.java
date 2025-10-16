package com.sparrowrecsys.online.datamanager;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

/**
 * 优化版RedisClient - 使用连接池
 */
public class OptimizedRedisClient {
    private static volatile JedisPool jedisPool;
    final static String REDIS_END_POINT = "localhost";
    final static int REDIS_PORT = 6379;

    private OptimizedRedisClient(){}

    public static JedisPool getPool(){
        if (null == jedisPool){
            synchronized (OptimizedRedisClient.class){
                if (null == jedisPool){
                    // 连接池配置
                    JedisPoolConfig config = new JedisPoolConfig();
                    config.setMaxTotal(20); // 最大连接数
                    config.setMaxIdle(10);  // 最大空闲连接
                    config.setMinIdle(5);   // 最小空闲连接
                    config.setTestOnBorrow(true);
                    config.setTestOnReturn(true);
                    
                    jedisPool = new JedisPool(config, REDIS_END_POINT, REDIS_PORT);
                }
            }
        }
        return jedisPool;
    }

    public static Jedis getInstance(){
        return getPool().getResource();
    }
}