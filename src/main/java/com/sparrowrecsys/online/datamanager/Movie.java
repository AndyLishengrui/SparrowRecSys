package com.sparrowrecsys.online.datamanager;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.sparrowrecsys.online.model.Embedding;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Movie 类，包含从movielens的movies.csv加载的属性和其他高级数据，如平均评分、嵌入向量等。
 */
public class Movie {
    int movieId; // 电影ID
    String title; // 电影标题
    int releaseYear; // 上映年份
    String imdbId; // IMDb ID
    String tmdbId; // TMDb ID
    List<String> genres; // 电影类型列表
    // 用户评分数量
    int ratingNumber;
    // 平均评分
    double averageRating;

    // 电影的嵌入向量
    @JsonIgnore
    Embedding emb;

    // 所有评分列表
    @JsonIgnore
    List<Rating> ratings;

    // 电影特征映射
    @JsonIgnore
    Map<String, String> movieFeatures;

    // 前10个评分的大小
    final int TOP_RATING_SIZE = 10;

    // 使用自定义序列化器序列化评分列表
    @JsonSerialize(using = RatingListSerializer.class)
    List<Rating> topRatings;

    // 构造函数，初始化默认值
    public Movie() {
        ratingNumber = 0;
        averageRating = 0;
        this.genres = new ArrayList<>();
        this.ratings = new ArrayList<>();
        this.topRatings = new LinkedList<>();
        this.emb = null;
        this.movieFeatures = null;
    }

    // 获取电影ID
    public int getMovieId() {
        return movieId;
    }

    // 设置电影ID
    public void setMovieId(int movieId) {
        this.movieId = movieId;
    }

    // 获取电影标题
    public String getTitle() {
        return title;
    }

    // 设置电影标题
    public void setTitle(String title) {
        this.title = title;
    }

    // 获取上映年份
    public int getReleaseYear() {
        return releaseYear;
    }

    // 设置上映年份
    public void setReleaseYear(int releaseYear) {
        this.releaseYear = releaseYear;
    }

    // 获取电影类型列表
    public List<String> getGenres() {
        return genres;
    }

    // 添加电影类型
    public void addGenre(String genre){
        this.genres.add(genre);
    }

    // 设置电影类型列表
    public void setGenres(List<String> genres) {
        this.genres = genres;
    }

    // 获取评分列表
    public List<Rating> getRatings() {
        return ratings;
    }

    // 添加评分并更新平均评分和评分数量
    public void addRating(Rating rating) {
        averageRating = (averageRating * ratingNumber + rating.getScore()) / (ratingNumber+1);
        ratingNumber++;
        this.ratings.add(rating);
        addTopRating(rating);
    }

    // 添加评分到前10个评分列表
    public void addTopRating(Rating rating){
        if (this.topRatings.isEmpty()){
            this.topRatings.add(rating);
        }else{
            int index = 0;
            for (Rating topRating : this.topRatings){
                if (topRating.getScore() >= rating.getScore()){
                    break;
                }
                index ++;
            }
            topRatings.add(index, rating);
            if (topRatings.size() > TOP_RATING_SIZE) {
                topRatings.remove(0);
            }
        }
    }

    // 获取IMDb ID
    public String getImdbId() {
        return imdbId;
    }

    // 设置IMDb ID
    public void setImdbId(String imdbId) {
        this.imdbId = imdbId;
    }

    // 获取TMDb ID
    public String getTmdbId() {
        return tmdbId;
    }

    // 设置TMDb ID
    public void setTmdbId(String tmdbId) {
        this.tmdbId = tmdbId;
    }

    // 获取评分数量
    public int getRatingNumber() {
        return ratingNumber;
    }

    // 获取平均评分
    public double getAverageRating() {
        return averageRating;
    }

    // 获取嵌入向量
    public Embedding getEmb() {
        return emb;
    }

    // 设置嵌入向量
    public void setEmb(Embedding emb) {
        this.emb = emb;
    }

    // 获取电影特征映射
    public Map<String, String> getMovieFeatures() {
        return movieFeatures;
    }

    // 设置电影特征映射
    public void setMovieFeatures(Map<String, String> movieFeatures) {
        this.movieFeatures = movieFeatures;
    }
}