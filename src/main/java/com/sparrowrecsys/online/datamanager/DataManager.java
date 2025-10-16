package com.sparrowrecsys.online.datamanager;

import com.sparrowrecsys.online.util.Config;
import com.sparrowrecsys.online.util.PerformanceMonitor;
import com.sparrowrecsys.online.util.Utility;
import redis.clients.jedis.Jedis;

import java.io.File;
import java.util.*;

/**
 * DataManager 是一个工具类，负责所有的数据加载逻辑。
 */
public class DataManager {
    // 单例实例
    private static volatile DataManager instance;
    // 存储电影数据的映射
    HashMap<Integer, Movie> movieMap;
    // 存储用户数据的映射
    HashMap<Integer, User> userMap;
    // 类型反向索引，用于快速查询某类型的所有电影
    HashMap<String, List<Movie>> genreReverseIndexMap;

    // 私有构造函数，初始化数据结构
    private DataManager(){
        this.movieMap = new HashMap<>();
        this.userMap = new HashMap<>();
        this.genreReverseIndexMap = new HashMap<>();
        instance = this;
    }

    // 获取单例实例
    public static DataManager getInstance(){
        if (null == instance){
            synchronized (DataManager.class){
                if (null == instance){
                    instance = new DataManager();
                }
            }
        }
        return instance;
    }

    // 从文件系统加载数据，包括电影、评分、链接数据和模型数据如嵌入向量
    public void loadData(String movieDataPath, String linkDataPath, String ratingDataPath, String movieEmbPath, String userEmbPath, String movieRedisKey, String userRedisKey) throws Exception{
        PerformanceMonitor.startTimer("DataManager.loadData");
        
        loadMovieData(movieDataPath);
        loadLinkData(linkDataPath);
        loadRatingData(ratingDataPath);
        loadMovieEmb(movieEmbPath, movieRedisKey);
        if (Config.IS_LOAD_ITEM_FEATURE_FROM_REDIS){
            loadMovieFeatures("mf:");
        }
        loadUserEmb(userEmbPath, userRedisKey);
        
        PerformanceMonitor.endTimer("DataManager.loadData");
        PerformanceMonitor.printReport();
    }

    // 重新加载embedding模型
    public void reloadEmbeddings(String modelDataPath, String movieEmbFile, String userEmbFile) throws Exception {
        String movieEmbPath = modelDataPath + movieEmbFile;
        String userEmbPath = modelDataPath + userEmbFile;
        
        System.out.println("重新加载embedding模型:");
        System.out.println("Movie embedding: " + movieEmbPath);
        System.out.println("User embedding: " + userEmbPath);
        
        // 清除现有的embedding数据
        for (Movie movie : movieMap.values()) {
            movie.setEmb(null);
        }
        for (User user : userMap.values()) {
            user.setEmb(null);
        }
        
        // 重新加载embedding
        loadMovieEmb(movieEmbPath, "i2vEmb");
        loadUserEmb(userEmbPath, "uEmb");
        
        System.out.println("✅ Embedding模型重新加载完成!");
    }

    // 从movies.csv加载电影数据
    private void loadMovieData(String movieDataPath) throws Exception{
        System.out.println("Loading movie data from " + movieDataPath + " ...");
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(movieDataPath))) {
            while (scanner.hasNextLine()) {
                String movieRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] movieData = movieRawData.split(",");
                if (movieData.length == 3){
                    Movie movie = new Movie();
                    movie.setMovieId(Integer.parseInt(movieData[0]));
                    int releaseYear = parseReleaseYear(movieData[1].trim());
                    if (releaseYear == -1){
                        movie.setTitle(movieData[1].trim());
                    }else{
                        movie.setReleaseYear(releaseYear);
                        movie.setTitle(movieData[1].trim().substring(0, movieData[1].trim().length()-6).trim());
                    }
                    String genres = movieData[2];
                    if (!genres.trim().isEmpty()){
                        String[] genreArray = genres.split("\\|");
                        for (String genre : genreArray){
                            movie.addGenre(genre);
                            addMovie2GenreIndex(genre, movie);
                        }
                    }
                    this.movieMap.put(movie.getMovieId(), movie);
                }
            }
        }
        System.out.println("Loading movie data completed. " + this.movieMap.size() + " movies in total.");
    }

    // 加载电影嵌入向量
    private void loadMovieEmb(String movieEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            PerformanceMonitor.startTimer("LoadMovieEmb_File");
            System.out.println("Loading movie embedding from " + movieEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(movieEmbPath))) {
                while (scanner.hasNextLine()) {
                    String movieRawEmbData = scanner.nextLine();
                    String[] movieEmbData = movieRawEmbData.split(":");
                    if (movieEmbData.length == 2) {
                        Movie m = getMovieById(Integer.parseInt(movieEmbData[0]));
                        if (null == m) {
                            continue;
                        }
                        m.setEmb(Utility.parseEmbStr(movieEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            PerformanceMonitor.endTimer("LoadMovieEmb_File");
            System.out.println("Loading movie embedding completed. " + validEmbCount + " movie embeddings in total.");
        } else {
            // 使用批量加载器优化Redis加载
            System.out.println("🚀 Using optimized batch loading for movie embeddings from Redis ...");
            int validEmbCount = BatchDataLoader.batchLoadMovieEmbeddings(this.movieMap, embKey + ":");
            System.out.println("✅ Optimized batch loading completed. " + validEmbCount + " movie embeddings loaded.");
        }
    }

    // 加载电影特征
    private void loadMovieFeatures(String movieFeaturesPrefix) throws Exception{
        System.out.println("🚀 Using optimized batch loading for movie features from Redis ...");
        int validFeaturesCount = BatchDataLoader.batchLoadMovieFeatures(this.movieMap, movieFeaturesPrefix);
        System.out.println("✅ Optimized batch loading completed. " + validFeaturesCount + " movie features loaded.");
    }

    // 加载用户嵌入向量
    private void loadUserEmb(String userEmbPath, String embKey) throws Exception{
        if (Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_FILE)) {
            PerformanceMonitor.startTimer("LoadUserEmb_File");
            System.out.println("Loading user embedding from " + userEmbPath + " ...");
            int validEmbCount = 0;
            try (Scanner scanner = new Scanner(new File(userEmbPath))) {
                while (scanner.hasNextLine()) {
                    String userRawEmbData = scanner.nextLine();
                    String[] userEmbData = userRawEmbData.split(":");
                    if (userEmbData.length == 2) {
                        // 支持两种用户ID格式: 纯数字或 "user_数字"
                        int userId;
                        String userIdStr = userEmbData[0];
                        if (userIdStr.startsWith("user_")) {
                            userId = Integer.parseInt(userIdStr.substring(5)); // 去掉"user_"前缀
                        } else {
                            userId = Integer.parseInt(userIdStr);
                        }
                        
                        User u = getUserById(userId);
                        if (null == u) {
                            continue;
                        }
                        u.setEmb(Utility.parseEmbStr(userEmbData[1]));
                        validEmbCount++;
                    }
                }
            }
            PerformanceMonitor.endTimer("LoadUserEmb_File");
            System.out.println("Loading user embedding completed. " + validEmbCount + " user embeddings in total.");
        } else {
            // 使用批量加载器优化Redis加载
            System.out.println("🚀 Using optimized batch loading for user embeddings from Redis ...");
            int validEmbCount = BatchDataLoader.batchLoadUserEmbeddings(this.userMap, embKey + ":");
            System.out.println("✅ Optimized batch loading completed. " + validEmbCount + " user embeddings loaded.");
        }
    }

    // 解析上映年份
    private int parseReleaseYear(String rawTitle){
        if (null == rawTitle || rawTitle.trim().length() < 6){
            return -1;
        } else {
            String yearString = rawTitle.trim().substring(rawTitle.length()-5, rawTitle.length()-1);
            try {
                return Integer.parseInt(yearString);
            } catch (NumberFormatException exception) {
                return -1;
            }
        }
    }

    // 从links.csv加载链接数据
    private void loadLinkData(String linkDataPath) throws Exception{
        System.out.println("Loading link data from " + linkDataPath + " ...");
        int count = 0;
        boolean skipFirstLine = true;
        try (Scanner scanner = new Scanner(new File(linkDataPath))) {
            while (scanner.hasNextLine()) {
                String linkRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = linkRawData.split(",");
                if (linkData.length == 3){
                    int movieId = Integer.parseInt(linkData[0]);
                    Movie movie = this.movieMap.get(movieId);
                    if (null != movie){
                        count++;
                        movie.setImdbId(linkData[1].trim());
                        movie.setTmdbId(linkData[2].trim());
                    }
                }
            }
        }
        System.out.println("Loading link data completed. " + count + " links in total.");
    }

    // 从ratings.csv加载评分数据
    private void loadRatingData(String ratingDataPath) throws Exception{
        System.out.println("Loading rating data from " + ratingDataPath + " ...");
        boolean skipFirstLine = true;
        int count = 0;
        try (Scanner scanner = new Scanner(new File(ratingDataPath))) {
            while (scanner.hasNextLine()) {
                String ratingRawData = scanner.nextLine();
                if (skipFirstLine){
                    skipFirstLine = false;
                    continue;
                }
                String[] linkData = ratingRawData.split(",");
                if (linkData.length == 4){
                    count++;
                    Rating rating = new Rating();
                    rating.setUserId(Integer.parseInt(linkData[0]));
                    rating.setMovieId(Integer.parseInt(linkData[1]));
                    rating.setScore(Float.parseFloat(linkData[2]));
                    rating.setTimestamp(Long.parseLong(linkData[3]));
                    Movie movie = this.movieMap.get(rating.getMovieId());
                    if (null != movie){
                        movie.addRating(rating);
                    }
                    if (!this.userMap.containsKey(rating.getUserId())){
                        User user = new User();
                        user.setUserId(rating.getUserId());
                        this.userMap.put(user.getUserId(), user);
                    }
                    this.userMap.get(rating.getUserId()).addRating(rating);
                }
            }
        }
        System.out.println("Loading rating data completed. " + count + " ratings in total.");
    }

    // 将电影添加到类型反向索引中
    private void addMovie2GenreIndex(String genre, Movie movie){
        if (!this.genreReverseIndexMap.containsKey(genre)){
            this.genreReverseIndexMap.put(genre, new ArrayList<>());
        }
        this.genreReverseIndexMap.get(genre).add(movie);
    }

    // 根据类型获取电影，并按sortBy方法排序
    public List<Movie> getMoviesByGenre(String genre, int size, String sortBy){
        if (null != genre){
            List<Movie> movies = new ArrayList<>(this.genreReverseIndexMap.get(genre));
            switch (sortBy){
                case "rating":
                    movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));
                    break;
                case "releaseYear":
                    movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));
                    break;
                default:
            }

            if (movies.size() > size){
                return movies.subList(0, size);
            }
            return movies;
        }
        return null;
    }

    // 获取前N部电影，并按sortBy方法排序
    public List<Movie> getMovies(int size, String sortBy){
        List<Movie> movies = new ArrayList<>(movieMap.values());
        switch (sortBy){
            case "rating":
                movies.sort((m1, m2) -> Double.compare(m2.getAverageRating(), m1.getAverageRating()));
                break;
            case "releaseYear":
                movies.sort((m1, m2) -> Integer.compare(m2.getReleaseYear(), m1.getReleaseYear()));
                break;
            default:
        }

        if (movies.size() > size){
            return movies.subList(0, size);
        }
        return movies;
    }

    // 根据电影ID获取电影对象
    public Movie getMovieById(int movieId){
        return this.movieMap.get(movieId);
    }

    // 根据用户ID获取用户对象
    public User getUserById(int userId){
        return this.userMap.get(userId);
    }
}