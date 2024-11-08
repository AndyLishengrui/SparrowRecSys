package com.sparrowrecsys.online.recprocess;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;
import java.util.*;

/**
 * 推荐相似电影的处理类
 */
public class SimilarMovieProcess {

    /**
     * 获取推荐电影列表
     * @param movieId 输入的电影ID
     * @param size 推荐电影的数量
     * @param model 用于计算相似度的模型
     * @return 相似电影列表
     */
    public static List<Movie> getRecList(int movieId, int size, String model){
        // 根据电影ID获取电影对象
        Movie movie = DataManager.getInstance().getMovieById(movieId);
        if (null == movie){
            return new ArrayList<>();
        }
        // 生成候选电影列表
        List<Movie> candidates = candidateGenerator(movie);
        // 对候选电影进行排序
        List<Movie> rankedList = ranker(movie, candidates, model);

        // 返回前size个推荐电影
        if (rankedList.size() > size){
            return rankedList.subList(0, size);
        }
        return rankedList;
    }

    /**
     * 生成相似电影的候选集
     * @param movie 输入的电影对象
     * @return 候选电影列表
     */
    public static List<Movie> candidateGenerator(Movie movie){
        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        // 根据电影类型生成候选集
        for (String genre : movie.getGenres()){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 100, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }
        // 移除输入的电影本身
        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * 多重检索候选集生成方法
     * @param movie 输入的电影对象
     * @return 候选电影列表
     */
    public static List<Movie> multipleRetrievalCandidates(Movie movie){
        if (null == movie){
            return null;
        }

        HashSet<String> genres = new HashSet<>(movie.getGenres());

        HashMap<Integer, Movie> candidateMap = new HashMap<>();
        // 根据电影类型生成候选集
        for (String genre : genres){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 20, "rating");
            for (Movie candidate : oneCandidates){
                candidateMap.put(candidate.getMovieId(), candidate);
            }
        }

        // 获取高评分电影作为候选集
        List<Movie> highRatingCandidates = DataManager.getInstance().getMovies(100, "rating");
        for (Movie candidate : highRatingCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        // 获取最新电影作为候选集
        List<Movie> latestCandidates = DataManager.getInstance().getMovies(100, "releaseYear");
        for (Movie candidate : latestCandidates){
            candidateMap.put(candidate.getMovieId(), candidate);
        }

        // 移除输入的电影本身
        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * 基于嵌入向量的候选集生成方法
     * @param movie 输入的电影对象
     * @param size 候选集的大小
     * @return 候选电影列表
     */
    public static List<Movie> retrievalCandidatesByEmbedding(Movie movie, int size){
        if (null == movie || null == movie.getEmb()){
            return null;
        }

        // 获取所有电影并计算相似度
        List<Movie> allCandidates = DataManager.getInstance().getMovies(10000, "rating");
        HashMap<Movie,Double> movieScoreMap = new HashMap<>();
        for (Movie candidate : allCandidates){
            double similarity = calculateEmbSimilarScore(movie, candidate);
            movieScoreMap.put(candidate, similarity);
        }

        // 对候选电影按相似度排序
        List<Map.Entry<Movie,Double>> movieScoreList = new ArrayList<>(movieScoreMap.entrySet());
        movieScoreList.sort(Map.Entry.comparingByValue());

        List<Movie> candidates = new ArrayList<>();
        for (Map.Entry<Movie,Double> movieScoreEntry : movieScoreList){
            candidates.add(movieScoreEntry.getKey());
        }

        return candidates.subList(0, Math.min(candidates.size(), size));
    }

    /**
     * 对候选电影进行排序
     * @param movie 输入的电影对象
     * @param candidates 候选电影列表
     * @param model 用于排序的模型名称
     * @return 排序后的电影列表
     */
    public static List<Movie> ranker(Movie movie, List<Movie> candidates, String model){
        HashMap<Movie, Double> candidateScoreMap = new HashMap<>();
        for (Movie candidate : candidates){
            double similarity;
            switch (model){
                case "emb":
                    // 使用嵌入向量计算相似度
                    similarity = calculateEmbSimilarScore(movie, candidate);
                    break;
                default:
                    // 使用默认方法计算相似度
                    similarity = calculateSimilarScore(movie, candidate);
            }
            candidateScoreMap.put(candidate, similarity);
        }
        // 对候选电影按相似度排序
        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m -> rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * 计算相似度得分
     * @param movie 输入的电影对象
     * @param candidate 候选电影
     * @return 相似度得分
     */
    public static double calculateSimilarScore(Movie movie, Movie candidate){
        int sameGenreCount = 0;
        // 计算相同类型的数量
        for (String genre : movie.getGenres()){
            if (candidate.getGenres().contains(genre)){
                sameGenreCount++;
            }
        }
        // 计算类型相似度
        double genreSimilarity = (double)sameGenreCount / (movie.getGenres().size() + candidate.getGenres().size()) / 2;
        // 计算评分得分
        double ratingScore = candidate.getAverageRating() / 5;

        double similarityWeight = 0.7;
        double ratingScoreWeight = 0.3;

        // 返回综合相似度得分
        return genreSimilarity * similarityWeight + ratingScore * ratingScoreWeight;
    }

    /**
     * 基于嵌入向量计算相似度得分
     * @param movie 输入的电影对象
     * @param candidate 候选电影
     * @return 相似度得分
     */
    public static double calculateEmbSimilarScore(Movie movie, Movie candidate){
        if (null == movie || null == candidate){
            return -1;
        }
        return movie.getEmb().calculateSimilarity(candidate.getEmb());
    }
}