package com.sparrowrecsys.online.model;

import java.util.ArrayList;

/**
 * Embedding 类，包含嵌入向量和相关计算
 */
public class Embedding {
    // 嵌入向量
    ArrayList<Float> embVector;

    // 默认构造函数，初始化空的嵌入向量
    public Embedding(){
        this.embVector = new ArrayList<>();
    }

    // 带参数的构造函数，使用给定的嵌入向量进行初始化
    public Embedding(ArrayList<Float> embVector){
        this.embVector = embVector;
    }

    // 添加一个维度到嵌入向量
    public void addDim(Float element){
        this.embVector.add(element);
    }

    // 获取嵌入向量
    public ArrayList<Float> getEmbVector() {
        return embVector;
    }

    // 设置嵌入向量
    public void setEmbVector(ArrayList<Float> embVector) {
        this.embVector = embVector;
    }

    // 计算两个嵌入向量之间的余弦相似度
    public double calculateSimilarity(Embedding otherEmb){
        // 检查向量是否为空或长度是否不一致
        if (null == embVector || null == otherEmb || null == otherEmb.getEmbVector()
                || embVector.size() != otherEmb.getEmbVector().size()){
            return -1; // 返回-1表示无法计算相似度
        }
        double dotProduct = 0; // 点积
        double denominator1 = 0; // 第一个向量的模
        double denominator2 = 0; // 第二个向量的模
        for (int i = 0; i < embVector.size(); i++){
            dotProduct += embVector.get(i) * otherEmb.getEmbVector().get(i); // 计算点积
            denominator1 += embVector.get(i) * embVector.get(i); // 计算第一个向量的模的平方
            denominator2 += otherEmb.getEmbVector().get(i) * otherEmb.getEmbVector().get(i); // 计算第二个向量的模的平方
        }
        // 返回余弦相似度
        return dotProduct / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
    }
}