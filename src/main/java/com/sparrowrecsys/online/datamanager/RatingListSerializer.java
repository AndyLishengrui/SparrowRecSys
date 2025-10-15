package com.sparrowrecsys.online.datamanager;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import java.io.IOException;
import java.util.List;

/**
 * RatingListSerializer 类，用于序列化 Rating 对象列表
 */
public class RatingListSerializer extends JsonSerializer<List<Rating>> {

    /**
     * 序列化方法，将 Rating 对象列表序列化为 JSON 数组
     * @param ratingList Rating 对象列表
     * @param jsonGenerator JSON 生成器
     * @param provider 序列化提供者
     * @throws IOException 如果发生 I/O 错误
     */
    @Override
    public void serialize(List<Rating> ratingList, JsonGenerator jsonGenerator,
                          SerializerProvider provider) throws IOException {
        jsonGenerator.writeStartArray(); // 开始写 JSON 数组
        for (Rating rating : ratingList) {
            jsonGenerator.writeStartObject(); // 开始写 JSON 对象
            jsonGenerator.writeObjectField("rating", rating); // 写入 rating 字段
            jsonGenerator.writeEndObject(); // 结束写 JSON 对象
        }
        jsonGenerator.writeEndArray(); // 结束写 JSON 数组
    }
}