package com.sparrowrecsys.online.util;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

/**
 * 性能监控工具类
 */
public class PerformanceMonitor {
    private static final Map<String, Long> startTimes = new ConcurrentHashMap<>();
    private static final Map<String, Long> totalTimes = new ConcurrentHashMap<>();
    private static final Map<String, Integer> callCounts = new ConcurrentHashMap<>();

    /**
     * 开始计时
     */
    public static void startTimer(String operationName) {
        startTimes.put(operationName, System.currentTimeMillis());
    }

    /**
     * 结束计时并记录
     */
    public static long endTimer(String operationName) {
        Long startTime = startTimes.remove(operationName);
        if (startTime == null) {
            System.err.println("⚠️ Warning: Timer not started for " + operationName);
            return -1;
        }

        long duration = System.currentTimeMillis() - startTime;
        
        // 累计总时间和调用次数
        totalTimes.merge(operationName, duration, Long::sum);
        callCounts.merge(operationName, 1, Integer::sum);

        System.out.println("⏱️ " + operationName + " completed in " + duration + "ms");
        return duration;
    }

    /**
     * 获取平均执行时间
     */
    public static double getAverageTime(String operationName) {
        Long totalTime = totalTimes.get(operationName);
        Integer count = callCounts.get(operationName);
        
        if (totalTime == null || count == null || count == 0) {
            return 0.0;
        }
        
        return (double) totalTime / count;
    }

    /**
     * 打印性能统计报告
     */
    public static void printReport() {
        System.out.println("\n📊 ========== 性能监控报告 ==========");
        for (String operation : totalTimes.keySet()) {
            Long totalTime = totalTimes.get(operation);
            Integer count = callCounts.get(operation);
            double avgTime = getAverageTime(operation);
            
            System.out.printf("🔹 %s:\n", operation);
            System.out.printf("   总调用次数: %d\n", count);
            System.out.printf("   总耗时: %dms\n", totalTime);
            System.out.printf("   平均耗时: %.2fms\n", avgTime);
            System.out.println();
        }
        System.out.println("=====================================\n");
    }

    /**
     * 清除所有统计数据
     */
    public static void reset() {
        startTimes.clear();
        totalTimes.clear();
        callCounts.clear();
    }
}