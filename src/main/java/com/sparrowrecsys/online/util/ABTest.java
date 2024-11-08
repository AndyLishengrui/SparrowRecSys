package com.sparrowrecsys.online.util;

/**
 * ABTest 类，用于进行A/B测试
 */
public class ABTest {
    // 流量分割数
    final static int trafficSplitNumber = 5;

    // A组模型名称
    final static String bucketAModel = "emb";
    // B组模型名称
    final static String bucketBModel = "nerualcf";

    // 默认模型名称
    final static String defaultModel = "emb";

    /**
     * 根据用户ID获取配置
     * @param userId 用户ID
     * @return 模型名称
     */
    public static String getConfigByUserId(String userId){
        // 如果用户ID为空或为空字符串，返回默认模型
        if (null == userId || userId.isEmpty()){
            return defaultModel;
        }

        // 根据用户ID的哈希值进行流量分割
        if(userId.hashCode() % trafficSplitNumber == 0){
            System.out.println(userId + " is in bucketA.");
            return bucketAModel; // A组模型
        }else if(userId.hashCode() % trafficSplitNumber == 1){
            System.out.println(userId + " is in bucketB.");
            return bucketBModel; // B组模型
        }else{
            System.out.println(userId + " isn't in AB test.");
            return defaultModel; // 默认模型
        }
    }
}