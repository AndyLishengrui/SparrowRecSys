package com.sparrowrecsys.online.util;

import com.sparrowrecsys.online.model.Embedding;

/**
 * Utility 类，提供实用工具方法
 */
public class Utility {

    /**
     * 将嵌入向量字符串解析为 Embedding 对象
     * @param embStr 嵌入向量字符串，元素之间用空格分隔
     * @return 解析后的 Embedding 对象
     */
    public static Embedding parseEmbStr(String embStr){
        // 将嵌入向量字符串按空格分隔成字符串数组
        String[] embStrings = embStr.split("\\s");
        // 创建一个新的 Embedding 对象
        Embedding emb = new Embedding();
        // 遍历字符串数组，将每个元素转换为浮点数并添加到嵌入向量中
        for (String element : embStrings) {
            emb.addDim(Float.parseFloat(element));
        }
        // 返回解析后的 Embedding 对象
        return emb;
    }
}