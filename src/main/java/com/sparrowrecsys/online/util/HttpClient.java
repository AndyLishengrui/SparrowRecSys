package com.sparrowrecsys.online.util;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.nio.client.CloseableHttpAsyncClient;
import org.apache.http.impl.nio.client.HttpAsyncClients;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Future;

/**
 * HttpClient 类，提供异步HTTP请求的功能
 */
public class HttpClient {

    /**
     * 发送异步POST请求
     * @param host 请求的主机地址
     * @param body 请求的内容
     * @return 响应内容
     */
    public static String asyncSinglePostRequest(String host, String body){
        if (null == body || body.isEmpty()){
            return null; // 如果请求内容为空，返回null
        }

        try {
            // 创建异步HTTP客户端
            final CloseableHttpAsyncClient client = HttpAsyncClients.createDefault();
            client.start(); // 启动客户端
            // 将请求内容转换为HttpEntity
            HttpEntity bodyEntity = new ByteArrayEntity(body.getBytes(StandardCharsets.UTF_8));
            // 创建POST请求
            HttpPost request = new HttpPost(host);
            request.setEntity(bodyEntity); // 设置请求内容
            // 发送请求并获取响应
            final Future<HttpResponse> future = client.execute(request, null);
            final HttpResponse response = future.get();
            client.close(); // 关闭客户端
            return getRespondContent(response); // 返回响应内容
        }catch (Exception e){
            e.printStackTrace();
            return "";
        }
    }

    /**
     * 发送多个异步POST请求
     * @param host 请求的主机地址
     * @param bodyMap 请求内容的映射
     * @return 响应内容的映射
     * @throws Exception 如果发生异常
     */
    public static Map<String, String> asyncMapPostRequest(String host, Map<String, String> bodyMap) throws Exception {
        if (null == bodyMap || bodyMap.isEmpty()){
            return null; // 如果请求内容映射为空，返回null
        }

        try {
            // 创建异步HTTP客户端
            final CloseableHttpAsyncClient client = HttpAsyncClients.createDefault();
            client.start(); // 启动客户端

            // 存储Future对象的映射
            HashMap<String, Future<HttpResponse>> futures = new HashMap<>();
            for (Map.Entry<String, String> bodyEntry : bodyMap.entrySet()) {
                String body = bodyEntry.getValue();
                HttpEntity bodyEntity = new ByteArrayEntity(body.getBytes(StandardCharsets.UTF_8));
                HttpPost request = new HttpPost(host);
                request.setEntity(bodyEntity);
                futures.put(bodyEntry.getKey(), client.execute(request, null));
            }

            // 存储响应内容的映射
            HashMap<String, String> responds = new HashMap<>();
            for (Map.Entry<String, Future<HttpResponse>> future : futures.entrySet()) {
                final HttpResponse response = future.getValue().get();
                responds.put(future.getKey(), getRespondContent(response));
            }

            client.close(); // 关闭客户端
            return responds; // 返回响应内容的映射
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 获取响应内容
     * @param response HTTP响应
     * @return 响应内容
     * @throws Exception 如果发生异常
     */
    public static String getRespondContent(HttpResponse response) throws Exception{
        HttpEntity entity = response.getEntity();
        InputStream is = entity.getContent();
        BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8), 8);
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null)
            sb.append(line).append("\n");
        return sb.toString();
    }

    public static void main(String[] args){
        // 创建第一个请求实例
        JSONObject instance = new JSONObject();
        instance.put("userId", 10351);
        instance.put("movieId", 52);

        // 创建第二个请求实例
        JSONObject instance2 = new JSONObject();
        instance2.put("userId", 10351);
        instance2.put("movieId", 53);

        // 将两个实例放入JSONArray
        JSONArray instances = new JSONArray();
        instances.put(instance);
        instances.put(instance2);

        // 创建请求的根对象
        JSONObject instancesRoot = new JSONObject();
        instancesRoot.put("instances", instances);

        // 打印请求内容
        System.out.println(instancesRoot.toString());

        // 发送异步POST请求并打印响应内容
        System.out.println(asyncSinglePostRequest("http://localhost:8501/v1/models/recmodel:predict", instancesRoot.toString()));
    }
}