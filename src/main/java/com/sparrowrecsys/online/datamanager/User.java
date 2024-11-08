package com.sparrowrecsys.online.datamanager;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.sparrowrecsys.online.model.Embedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * User 类，包含从 movielens 的 ratings.csv 加载的属性
 */
public class User {
    int userId; // 用户ID
    double averageRating = 0; // 平均评分
    double highestRating = 0; // 最高评分
    double lowestRating = 5.0; // 最低评分
    int ratingCount = 0; // 评分数量

    @JsonSerialize(using = RatingListSerializer.class)
    List<Rating> ratings; // 用户的评分列表

    // 用户的嵌入向量
    @JsonIgnore
    Embedding emb;

    // 用户特征映射
    @JsonIgnore
    Map<String, String> userFeatures;

    // 构造函数，初始化默认值
    public User() {
        this.ratings = new ArrayList<>();
        this.emb = null;
        this.userFeatures = null;
    }

    // 获取用户ID
    public int getUserId() {
        return userId;
    }

    // 设置用户ID
    public void setUserId(int userId) {
        this.userId = userId;
    }

    // 获取用户的评分列表
    public List<Rating> getRatings() {
        return ratings;
    }

    // 设置用户的评分列表
    public void setRatings(List<Rating> ratings) {
        this.ratings = ratings;
    }

    // 添加评分并更新平均评分、最高评分、最低评分和评分数量
    public void addRating(Rating rating) {
        this.ratings.add(rating);
        this.averageRating = (this.averageRating * ratingCount + rating.getScore()) / (ratingCount + 1);
        if (rating.getScore() > highestRating) {
            highestRating = rating.getScore();
        }

        if (rating.getScore() < lowestRating) {
            lowestRating = rating.getScore();
        }

        ratingCount++;
    }

    // 获取平均评分
    public double getAverageRating() {
        return averageRating;
    }

    // 设置平均评分
    public void setAverageRating(double averageRating) {
        this.averageRating = averageRating;
    }

    // 获取最高评分
    public double getHighestRating() {
        return highestRating;
    }

    // 设置最高评分
    public void setHighestRating(double highestRating) {
        this.highestRating = highestRating;
    }

    // 获取最低评分
    public double getLowestRating() {
        return lowestRating;
    }

    // 设置最低评分
    public void setLowestRating(double lowestRating) {
        this.lowestRating = lowestRating;
    }

    // 获取评分数量
    public int getRatingCount() {
        return ratingCount;
    }

    // 设置评分数量
    public void setRatingCount(int ratingCount) {
        this.ratingCount = ratingCount;
    }

    // 获取嵌入向量
    public Embedding getEmb() {
        return emb;
    }

    // 设置嵌入向量
    public void setEmb(Embedding emb) {
        this.emb = emb;
    }

    // 获取用户特征映射
    public Map<String, String> getUserFeatures() {
        return userFeatures;
    }

    // 设置用户特征映射
    public void setUserFeatures(Map<String, String> userFeatures) {
        this.userFeatures = userFeatures;
    }
}