# 实验4：Redis集成加速推荐系统

## 实验概述

本实验将引导学生为SparrowRecSys推荐系统集成Redis缓存，显著提升系统的数据访问性能。通过本实验，学生将学习如何：

1. 分析现有Redis集成架构
2. 安装和配置Redis服务器
3. 实现数据从文件系统到Redis的迁移
4. 验证Redis集成效果和性能提升

## 实验目标

- **技术目标**：掌握Redis在推荐系统中的应用
- **性能目标**：提升数据访问速度，优化系统响应时间
- **架构目标**：理解缓存层在推荐系统中的重要作用

## 预备知识

- Java编程基础
- Maven项目管理
- 推荐系统基本概念
- Redis基本原理和操作

## 实验环境

- **操作系统**：macOS/Linux/Windows
- **Java版本**：JDK 8+
- **Maven版本**：3.6+
- **Redis版本**：7.0+
- **IDE**：VSCode/IntelliJ IDEA

## 实验步骤

### 步骤1：分析现有Redis集成架构

#### 1.1 检查Redis客户端实现

查看 `src/main/java/com/sparrowrecsys/online/datamanager/RedisClient.java`：

```java
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
```

#### 1.2 检查配置开关

查看 `src/main/java/com/sparrowrecsys/online/util/Config.java`：

```java
package com.sparrowrecsys.online.util;

public class Config {
    // 数据源类型常量
    public static final String DATA_SOURCE_REDIS = "redis";
    public static final String DATA_SOURCE_FILE = "file";

    // 嵌入向量的数据源配置
    public static String EMB_DATA_SOURCE = Config.DATA_SOURCE_FILE; // 默认文件
    // Redis特征加载配置
    public static boolean IS_LOAD_USER_FEATURE_FROM_REDIS = false;
    public static boolean IS_LOAD_ITEM_FEATURE_FROM_REDIS = false;
}
```

#### 1.3 分析数据加载逻辑

在 `DataManager.java` 中找到双重数据源支持：

```java
private void loadMovieEmb(String movieEmbPath, String embKey) throws Exception{
    if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
        // 从文件加载
        System.out.println("Loading movie embedding from " + movieEmbPath + " ...");
        // ... 文件读取逻辑
    } else {
        // 从Redis加载
        System.out.println("Loading movie embedding from Redis ...");
        Set<String> movieEmbKeys = RedisClient.getInstance().keys(embKey + "*");
        // ... Redis读取逻辑
    }
}
```

### 步骤2：安装和配置Redis服务器

#### 2.1 安装Redis

**macOS用户（使用Homebrew）：**
```bash
brew install redis
```

**Linux用户（从源码编译）：**
```bash
# 创建临时目录
mkdir -p ~/redis_temp && cd ~/redis_temp

# 下载Redis源码
curl -O https://download.redis.io/redis-stable.tar.gz

# 解压并编译
tar xzf redis-stable.tar.gz
cd redis-stable
make
```

#### 2.2 启动Redis服务器

**使用Homebrew安装的Redis：**
```bash
redis-server
```

**使用源码编译的Redis：**
```bash
~/redis_temp/redis-stable/src/redis-server --daemonize yes
```

#### 2.3 验证Redis连接

```bash
# 测试连接
redis-cli ping
# 期望输出: PONG

# 测试基本操作
redis-cli set test_key "Hello Redis"
redis-cli get test_key
# 期望输出: "Hello Redis"
```

### 步骤3：创建数据迁移工具

#### 3.1 创建RedisDataMigrator类

创建文件 `src/main/java/com/sparrowrecsys/online/util/RedisDataMigrator.java`：

```java
package com.sparrowrecsys.online.util;

import com.sparrowrecsys.online.datamanager.RedisClient;
import redis.clients.jedis.Jedis;
import java.io.File;
import java.util.Scanner;
import java.util.Set;

/**
 * Redis数据迁移工具
 * 将嵌入向量和特征数据从文件系统迁移到Redis存储
 */
public class RedisDataMigrator {
    
    /**
     * 将电影嵌入向量迁移到Redis
     */
    public static void migrateMovieEmbeddings(String movieEmbPath) throws Exception {
        System.out.println("开始迁移电影嵌入向量到Redis...");
        Jedis redisClient = RedisClient.getInstance();
        int migratedCount = 0;
        
        try (Scanner scanner = new Scanner(new File(movieEmbPath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(":", 2);
                if (parts.length == 2) {
                    String movieId = parts[0];
                    String embedding = parts[1];
                    String redisKey = "i2vEmb:" + movieId;
                    
                    redisClient.set(redisKey, embedding);
                    migratedCount++;
                    
                    if (migratedCount % 100 == 0) {
                        System.out.println("已迁移 " + migratedCount + " 个电影嵌入向量");
                    }
                }
            }
        }
        System.out.println("电影嵌入向量迁移完成！总共迁移了 " + migratedCount + " 个向量");
    }
    
    /**
     * 将用户嵌入向量迁移到Redis
     */
    public static void migrateUserEmbeddings(String userEmbPath) throws Exception {
        System.out.println("开始迁移用户嵌入向量到Redis...");
        Jedis redisClient = RedisClient.getInstance();
        int migratedCount = 0;
        
        try (Scanner scanner = new Scanner(new File(userEmbPath))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(":", 2);
                if (parts.length == 2) {
                    String userId = parts[0];
                    String embedding = parts[1];
                    String redisKey = "uEmb:" + userId;
                    
                    redisClient.set(redisKey, embedding);
                    migratedCount++;
                    
                    if (migratedCount % 100 == 0) {
                        System.out.println("已迁移 " + migratedCount + " 个用户嵌入向量");
                    }
                }
            }
        }
        System.out.println("用户嵌入向量迁移完成！总共迁移了 " + migratedCount + " 个向量");
    }
    
    /**
     * 检查Redis中的数据统计
     */
    public static void checkRedisDataStats() throws Exception {
        Jedis redisClient = RedisClient.getInstance();
        
        Set<String> movieEmbKeys = redisClient.keys("i2vEmb:*");
        Set<String> userEmbKeys = redisClient.keys("uEmb:*");
        
        System.out.println("Redis数据统计:");
        System.out.println("电影嵌入向量数量: " + movieEmbKeys.size());
        System.out.println("用户嵌入向量数量: " + userEmbKeys.size());
    }
    
    public static void main(String[] args) {
        try {
            String movieEmbPath = "src/main/resources/webroot/modeldata/item2vecEmb.csv";
            String userEmbPath = "src/main/resources/webroot/modeldata/userEmb.csv";
            
            System.out.println("=== Redis数据迁移工具 ===");
            
            // 测试Redis连接
            System.out.println("测试Redis连接...");
            Jedis redisClient = RedisClient.getInstance();
            redisClient.set("test", "migration_test");
            String testResult = redisClient.get("test");
            if ("migration_test".equals(testResult)) {
                System.out.println("✅ Redis连接正常");
                redisClient.del("test");
            } else {
                System.out.println("❌ Redis连接失败");
                return;
            }
            
            // 执行数据迁移
            migrateMovieEmbeddings(movieEmbPath);
            migrateUserEmbeddings(userEmbPath);
            
            // 显示统计信息
            System.out.println("\n=== 迁移完成统计 ===");
            checkRedisDataStats();
            
            System.out.println("\n✅ 数据迁移完成！现在可以修改Config.java启用Redis数据源");
            
        } catch (Exception e) {
            System.err.println("数据迁移失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
```

#### 3.2 编译项目

```bash
mvn compile
```

#### 3.3 运行数据迁移

```bash
mvn exec:java -Dexec.mainClass="com.sparrowrecsys.online.util.RedisDataMigrator"
```

预期输出：
```
=== Redis数据迁移工具 ===
测试Redis连接...
✅ Redis连接正常
开始迁移电影嵌入向量到Redis...
已迁移 100 个电影嵌入向量
...
电影嵌入向量迁移完成！总共迁移了 881 个向量
开始迁移用户嵌入向量到Redis...
已迁移 100 个用户嵌入向量
...
用户嵌入向量迁移完成！总共迁移了 29776 个向量

=== 迁移完成统计 ===
Redis数据统计:
电影嵌入向量数量: 881
用户嵌入向量数量: 29776

✅ 数据迁移完成！现在可以修改Config.java启用Redis数据源
```

### 步骤4：启用Redis数据源

#### 4.1 修改配置文件

编辑 `src/main/java/com/sparrowrecsys/online/util/Config.java`：

```java
// 修改前：
public static String EMB_DATA_SOURCE = Config.DATA_SOURCE_FILE;
public static boolean IS_LOAD_USER_FEATURE_FROM_REDIS = false;
public static boolean IS_LOAD_ITEM_FEATURE_FROM_REDIS = false;

// 修改后：
public static String EMB_DATA_SOURCE = Config.DATA_SOURCE_REDIS;
public static boolean IS_LOAD_USER_FEATURE_FROM_REDIS = true;
public static boolean IS_LOAD_ITEM_FEATURE_FROM_REDIS = true;
```

#### 4.2 重新编译和启动系统

```bash
# 编译项目
mvn compile

# 启动推荐系统
mvn exec:java -Dexec.mainClass="com.sparrowrecsys.online.RecSysServer"
```

预期启动日志：
```
Loading movie data from ...movies.csv ...
Loading movie data completed. 755 movies in total.
Loading rating data from ...ratings.csv ...
Loading rating data completed. 1168638 ratings in total.
Loading movie embedding from Redis ...
Loading movie embedding completed. 675 movie embeddings in total.
RecSys 服务器已启动。
```

### 步骤5：验证Redis集成效果

#### 5.1 测试系统功能

```bash
# 测试网站首页
curl -I "http://localhost:6010/"
# 期望输出: HTTP/1.1 200 OK

# 测试模型列表API
curl -s "http://localhost:6010/getmodel?action=list"
# 期望输出: JSON格式的模型列表

# 测试推荐API
curl -s "http://localhost:6010/rec?userId=1&size=5"
# 期望输出: JSON格式的推荐结果
```

#### 5.2 性能对比分析

**文件系统模式特点：**
- 每次启动需要解析CSV文件
- I/O操作受限于磁盘性能
- 数据访问延迟较高

**Redis模式特点：**
- 数据预加载到内存
- 快速键值对访问
- 显著提升响应速度

## 实验结果

成功完成本实验后，您将获得：

1. **完整的Redis集成架构**：理解Redis在推荐系统中的作用
2. **数据迁移能力**：掌握从文件系统到Redis的数据迁移方法
3. **性能优化经验**：体验缓存对系统性能的显著提升
4. **企业级架构**：学习真实项目中的缓存应用模式

## 数据存储格式

### Redis键值对结构

- **电影嵌入向量**：`i2vEmb:movieId` → `embedding_vector_string`
- **用户嵌入向量**：`uEmb:userId` → `embedding_vector_string`
- **电影特征**：`movieFeatures:movieId` → `Hash存储`

### 示例数据

```redis
# 电影嵌入向量
i2vEmb:1 → "-0.1234 0.5678 -0.9012 ..."

# 用户嵌入向量  
uEmb:123 → "0.2345 -0.6789 0.1234 ..."
```

## 常见问题和解决方案

### Q1: Redis连接失败
**问题**：无法连接到Redis服务器
**解决方案**：
1. 检查Redis服务是否启动：`ps aux | grep redis-server`
2. 检查端口是否被占用：`lsof -i:6379`
3. 重启Redis服务

### Q2: 数据迁移失败
**问题**：迁移过程中出现异常
**解决方案**：
1. 检查文件路径是否正确
2. 确认Redis服务正常运行
3. 检查内存是否充足

### Q3: 系统启动后仍从文件加载
**问题**：配置未生效
**解决方案**：
1. 确认Config.java修改正确
2. 重新编译项目：`mvn clean compile`
3. 检查启动日志确认数据源

## 扩展实验

1. **性能基准测试**：对比文件系统与Redis的响应时间
2. **大数据集测试**：使用更大的数据集测试Redis性能
3. **Redis集群**：配置Redis集群以支持更大规模数据
4. **数据持久化**：配置Redis数据持久化策略

## 实验小结

通过本实验，学生深入理解了：
- Redis在推荐系统中的重要作用
- 缓存层的架构设计原理
- 数据迁移的实践方法
- 性能优化的实际效果

这为学生未来从事大规模推荐系统开发奠定了坚实的技术基础。