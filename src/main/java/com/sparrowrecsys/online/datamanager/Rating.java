package com.sparrowrecsys.online.datamanager;

/**
 * Rating 类，包含从movielens的ratings.csv加载的属性
 */
public class Rating {
    int movieId; // 电影ID
    int userId; // 用户ID
    float score; // 评分
    long timestamp; // 时间戳

    // 获取电影ID
    public int getMovieId() {
        return movieId;
    }

    // 设置电影ID
    public void setMovieId(int movieId) {
        this.movieId = movieId;
    }

    // 获取用户ID
    public int getUserId() {
        return userId;
    }

    // 设置用户ID
    public void setUserId(int userId) {
        this.userId = userId;
    }

    // 获取评分
    public float getScore() {
        return score;
    }

    // 设置评分
    public void setScore(float score) {
        this.score = score;
    }

    // 获取时间戳
    public long getTimestamp() {
        return timestamp;
    }

    // 设置时间戳
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
}