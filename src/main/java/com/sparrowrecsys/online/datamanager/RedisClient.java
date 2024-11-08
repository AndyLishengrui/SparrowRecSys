package com.sparrowrecsys.online.datamanager;

import redis.clients.jedis.Jedis;

/**
 * RedisClient 类，提供 Redis 客户端的单例模式
 */
public class RedisClient {
    // 单例模式的 Jedis 客户端
    private static volatile Jedis redisClient;
    // Redis 服务器地址
    final static String REDIS_END_POINT = "localhost";
    // Redis 服务器端口
    final static int REDIS_PORT = 6379;

    // 私有构造函数，初始化 Jedis 客户端
    private RedisClient(){
        redisClient = new Jedis(REDIS_END_POINT, REDIS_PORT);
    }

    /**
     * 获取 Jedis 客户端的单例实例
     * @return Jedis 客户端实例
     */
    public static Jedis getInstance(){
        if (null == redisClient){
            synchronized (RedisClient.class){
                if (null == redisClient){
                    redisClient = new Jedis(REDIS_END_POINT, REDIS_PORT);
                }
            }
        }
        return redisClient;
    }
}