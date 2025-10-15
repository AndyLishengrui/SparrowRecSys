package com.sparrowrecsys.online.util;

/**
 * Config 类，包含系统配置常量和变量
 */
public class Config {
    // 数据源类型常量：Redis
    public static final String DATA_SOURCE_REDIS = "redis";
    // 数据源类型常量：文件
    public static final String DATA_SOURCE_FILE = "file";

    // 嵌入向量的数据源，启用Redis加速
    public static String EMB_DATA_SOURCE = Config.DATA_SOURCE_REDIS;
    // 是否从Redis加载用户特征，启用加速
    public static boolean IS_LOAD_USER_FEATURE_FROM_REDIS = true;
    // 是否从Redis加载物品特征，启用加速
    public static boolean IS_LOAD_ITEM_FEATURE_FROM_REDIS = true;

    // 是否启用A/B测试，默认是false
    public static boolean IS_ENABLE_AB_TEST = false;
}