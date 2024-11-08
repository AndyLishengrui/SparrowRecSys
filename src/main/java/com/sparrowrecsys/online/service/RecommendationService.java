package com.sparrowrecsys.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

/**
 * RecommendationService 类，提供基于不同输入的推荐服务
 */
public class RecommendationService extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws ServletException,
            IOException {
        try {
            // 设置响应内容类型为JSON
            response.setContentType("application/json");
            // 设置响应状态为200 OK
            response.setStatus(HttpServletResponse.SC_OK);
            // 设置响应字符编码为UTF-8
            response.setCharacterEncoding("UTF-8");
            // 设置允许跨域访问
            response.setHeader("Access-Control-Allow-Origin", "*");

            // 获取电影类型参数
            String genre = request.getParameter("genre");
            // 获取返回电影数量参数
            String size = request.getParameter("size");
            // 获取排序算法参数
            String sortby = request.getParameter("sortby");

            // 根据电影类型、数量和排序算法获取电影列表
            List<Movie> movies = DataManager.getInstance().getMoviesByGenre(genre, Integer.parseInt(size), sortby);

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