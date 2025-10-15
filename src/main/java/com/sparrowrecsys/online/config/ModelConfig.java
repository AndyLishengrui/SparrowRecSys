package com.sparrowrecsys.online.config;

/**
 * 模型配置类
 */
public class ModelConfig {
    // 可用的模型版本
    public enum ModelVersion {
        STANDARD("标准版本", "item2vecEmb.csv", "userEmb.csv"),
        LARGE("大数据集版本", "item2vecEmb_large.csv", "userEmb_large.csv");
        
        private final String displayName;
        private final String itemEmbFile;
        private final String userEmbFile;
        
        ModelVersion(String displayName, String itemEmbFile, String userEmbFile) {
            this.displayName = displayName;
            this.itemEmbFile = itemEmbFile;
            this.userEmbFile = userEmbFile;
        }
        
        public String getDisplayName() {
            return displayName;
        }
        
        public String getItemEmbFile() {
            return itemEmbFile;
        }
        
        public String getUserEmbFile() {
            return userEmbFile;
        }
    }
    
    // 当前使用的模型版本
    private static ModelVersion currentModelVersion = ModelVersion.STANDARD;
    
    public static ModelVersion getCurrentModelVersion() {
        return currentModelVersion;
    }
    
    public static void setCurrentModelVersion(ModelVersion version) {
        currentModelVersion = version;
    }
    
    public static String getCurrentItemEmbFile() {
        return currentModelVersion.getItemEmbFile();
    }
    
    public static String getCurrentUserEmbFile() {
        return currentModelVersion.getUserEmbFile();
    }
    
    public static ModelVersion[] getAllModelVersions() {
        return ModelVersion.values();
    }
}
