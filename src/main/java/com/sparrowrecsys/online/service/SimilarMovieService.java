package com.sparrowrecsys.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.datamanager.Movie;
import com.sparrowrecsys.online.recprocess.SimilarMovieProcess;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

/**
 * SimilarMovieService 类，根据特定电影推荐相似电影
 */
public class SimilarMovieService extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            // 设置响应内容类型为JSON
            response.setContentType("application/json");
            // 设置响应状态为200 OK
            response.setStatus(HttpServletResponse.SC_OK);
            // 设置响应字符编码为UTF-8
            response.setCharacterEncoding("UTF-8");
            // 设置允许跨域访问
            response.setHeader("Access-Control-Allow-Origin", "*");

            // 获取电影ID参数
            String movieId = request.getParameter("movieId");
            // 获取返回电影数量参数
            String size = request.getParameter("size");
            // 获取计算相似度的模型参数，例如embedding, graph-embedding
            String model = request.getParameter("model");

            // 使用 SimilarMovieProcess 获取相似电影列表
            List<Movie> movies = SimilarMovieProcess.getRecList(Integer.parseInt(movieId), Integer.parseInt(size), model);

            // 将电影列表转换为JSON格式并返回
            ObjectMapper mapper = new ObjectMapper();
            String jsonMovies = mapper.writeValueAsString(movies);
            response.getWriter().println(jsonMovies);

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}