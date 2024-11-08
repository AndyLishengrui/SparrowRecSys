package com.sparrowrecsys.nearline.flink;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.io.TextInputFormat;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.source.FileProcessingMode;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.net.URL;

// 定义Rating类，用于存储评分数据
class Rating {
    public String userId;
    public String movieId;
    public String rating;
    public String timestamp;
    public String latestMovieId;

    // 构造函数，通过解析CSV行初始化Rating对象
    public Rating(String line) {
        String[] lines = line.split(",");
        this.userId = lines[0];
        this.movieId = lines[1];
        this.rating = lines[2];
        this.timestamp = lines[3];
        this.latestMovieId = lines[1];
    }
}

public class RealTimeFeature {

    public void test() throws Exception {
        // 设置执行环境
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 获取评分数据的路径
        URL ratingResourcesPath = this.getClass().getResource("/webroot/sampledata/ratings.csv");

        // 监控目录，检查新文件
        TextInputFormat format = new TextInputFormat(new org.apache.flink.core.fs.Path(ratingResourcesPath.getPath()));

        // 读取文件数据流
        DataStream<String> inputStream = env.readFile(
                format,
                ratingResourcesPath.getPath(),
                FileProcessingMode.PROCESS_CONTINUOUSLY,
                100);

        // 将输入流映射为Rating对象流
        DataStream<Rating> ratingStream = inputStream.map(Rating::new);

        // 按userId键分组，设置时间窗口为1秒，并进行reduce操作
        ratingStream.keyBy(rating -> rating.userId)
                .timeWindow(Time.seconds(1))
                .reduce(
                        (ReduceFunction<Rating>) (rating, t1) -> {
                            // 比较时间戳，保留最新的Rating对象
                            if (rating.timestamp.compareTo(t1.timestamp) > 0) {
                                return rating;
                            } else {
                                return t1;
                            }
                        }
                ).addSink(new SinkFunction<Rating>() {
            @Override
            public void invoke(Rating value, Context context) {
                // 输出最新的电影ID
                System.out.println("userId:" + value.userId + "\tlatestMovieId:" + value.latestMovieId);
            }
        });

        // 执行Flink作业
        env.execute();
    }

    public static void main(String[] args) throws Exception {
        // 运行测试方法
        new RealTimeFeature().test();
    }
}