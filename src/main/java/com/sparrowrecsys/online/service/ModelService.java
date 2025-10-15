package com.sparrowrecsys.online.service;

import com.sparrowrecsys.online.config.ModelConfig;
import com.sparrowrecsys.online.datamanager.DataManager;
import org.json.JSONArray;
import org.json.JSONObject;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * 模型管理服务
 */
public class ModelService extends HttpServlet {
    
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");
        response.setHeader("Access-Control-Allow-Origin", "*");
        
        PrintWriter out = response.getWriter();
        JSONObject responseJson = new JSONObject();
        
        try {
            String action = request.getParameter("action");
            
            if ("list".equals(action)) {
                // 获取所有可用的模型版本
                JSONArray modelsArray = new JSONArray();
                for (ModelConfig.ModelVersion version : ModelConfig.getAllModelVersions()) {
                    JSONObject modelJson = new JSONObject();
                    modelJson.put("version", version.name());
                    modelJson.put("displayName", version.getDisplayName());
                    modelJson.put("itemEmbFile", version.getItemEmbFile());
                    modelJson.put("userEmbFile", version.getUserEmbFile());
                    modelJson.put("isCurrent", version == ModelConfig.getCurrentModelVersion());
                    modelsArray.put(modelJson);
                }
                
                responseJson.put("success", true);
                responseJson.put("models", modelsArray);
                responseJson.put("currentModel", ModelConfig.getCurrentModelVersion().name());
                
            } else if ("switch".equals(action)) {
                // 切换模型版本
                String versionName = request.getParameter("version");
                if (versionName != null) {
                    try {
                        ModelConfig.ModelVersion newVersion = ModelConfig.ModelVersion.valueOf(versionName);
                        
                        // 获取模型文件路径
                        String webRoot = getServletContext().getRealPath("/");
                        if (webRoot == null) {
                            // 如果getRealPath返回null，尝试使用类路径
                            webRoot = this.getClass().getResource("/webroot/").getPath();
                        }
                        
                        // 确保路径以/结尾
                        if (!webRoot.endsWith("/")) {
                            webRoot += "/";
                        }
                        String modelDataPath = webRoot + "modeldata/";
                        
                        // 重新加载embedding
                        DataManager.getInstance().reloadEmbeddings(
                            modelDataPath,
                            newVersion.getItemEmbFile(),
                            newVersion.getUserEmbFile()
                        );
                        
                        // 更新当前模型版本
                        ModelConfig.setCurrentModelVersion(newVersion);
                        
                        responseJson.put("success", true);
                        responseJson.put("message", "成功切换到" + newVersion.getDisplayName());
                        responseJson.put("currentModel", newVersion.name());
                        
                    } catch (IllegalArgumentException e) {
                        responseJson.put("success", false);
                        responseJson.put("message", "无效的模型版本: " + versionName);
                        e.printStackTrace();
                    } catch (Exception e) {
                        responseJson.put("success", false);
                        responseJson.put("message", "模型切换失败: " + e.getMessage());
                        e.printStackTrace();
                    }
                } else {
                    responseJson.put("success", false);
                    responseJson.put("message", "未指定模型版本");
                }
                
            } else {
                responseJson.put("success", false);
                responseJson.put("message", "无效的操作: " + action);
            }
            
        } catch (Exception e) {
            responseJson.put("success", false);
            responseJson.put("message", "服务器错误: " + e.getMessage());
            e.printStackTrace();
        }
        
        out.println(responseJson.toString());
        out.close();
    }
    
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        doGet(request, response);
    }
}
