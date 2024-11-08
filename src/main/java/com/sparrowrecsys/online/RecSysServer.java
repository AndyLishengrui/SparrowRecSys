//   这段代码是一个基于 Java 和 Jetty 框架的 Web 服务器，主要用于提供在线推荐系统的服务。以下是对代码的详细分析：

// 包和导入：

// package com.sparrowrecsys.online;：声明了代码所在的包。
// 导入了多个类和包，包括数据管理、服务、Jetty 服务器、Servlet 相关类等。
// 类声明：

// public class RecSysServer：声明了一个名为 RecSysServer 的公共类。
// 主方法：

// public static void main(String[] args) throws Exception：程序的入口点，创建了一个 RecSysServer 实例并调用其 run 方法。
// 默认端口：

// private static final int DEFAULT_PORT = 6010;：定义了一个私有静态常量 DEFAULT_PORT，值为 6010，作为服务器的默认端口。
// 运行方法：

// public void run() throws Exception：RecSysServer 类的核心方法，负责启动服务器。
// 端口设置：尝试从环境变量中获取端口号，如果获取失败则使用默认端口。
// 服务器初始化：创建一个 Server 对象，并设置其监听地址和端口。
// 获取 Web 根目录：通过类加载器获取 index.html 文件的路径。
// 设置欢迎页面：将 index.html 设置为服务器的欢迎页面。
// 加载数据：调用 DataManager 的 loadData 方法加载数据。
// 创建 Servlet 上下文：创建一个 ServletContextHandler 对象，并设置上下文路径、资源路径和欢迎文件。
// 添加 Servlet：添加多个 Servlet，包括默认 Servlet 和自定义的服务 Servlet。
// 设置服务器处理器：将创建的 ServletContextHandler 设置为服务器的处理器。
// 启动服务器：启动服务器，并等待其结束。
// 异常处理：

// 使用 try-catch 块捕获并处理可能发生的异常。
// 总结来说，这段代码的主要功能是启动一个 Web 服务器，加载数据，并提供多个服务接口，如获取电影信息、用户信息、相似电影推荐、个性化推荐等。
//  */
package com.sparrowrecsys.online;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.service.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

/***
 * Recsys 服务器，在线推荐服务的端点
 */

public class RecSysServer {

    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }

    // recsys 服务器端口号
    private static final int DEFAULT_PORT = 6010;

    public void run() throws Exception {

        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        // 设置 IP 和端口号
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        // 获取 index.html 路径
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null) {
            throw new IllegalStateException("无法确定 Web 根目录 URL 位置");
        }

        // 将 index.html 设置为根页面
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$", "/"));
        System.out.printf("Web 根目录 URI: %s%n", webRootUri.getPath());

        // 将所有数据加载到 DataManager
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata/movies.csv",
                webRootUri.getPath() + "sampledata/links.csv", webRootUri.getPath() + "sampledata/ratings.csv",
                webRootUri.getPath() + "modeldata/item2vecEmb.csv",
                webRootUri.getPath() + "modeldata/userEmb.csv",
                "i2vEmb", "uEmb");

        // 创建服务器上下文
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[] { "index.html" });
        context.getMimeTypes().addMimeMapping("txt", "text/plain;charset=utf-8");

        // 将服务与不同的 Servlet 绑定
        context.addServlet(DefaultServlet.class, "/");
        context.addServlet(new ServletHolder(new MovieService()), "/getmovie");
        context.addServlet(new ServletHolder(new UserService()), "/getuser");
        context.addServlet(new ServletHolder(new SimilarMovieService()), "/getsimilarmovie");
        context.addServlet(new ServletHolder(new RecommendationService()), "/getrecommendation");
        context.addServlet(new ServletHolder(new RecForYouService()), "/getrecforyou");

        // 设置 URL 处理器
        server.setHandler(context);
        System.out.println("RecSys 服务器已启动。");

        // 启动服务器
        server.start();
        server.join();
    }
}
