package com.sparrowrecsys.offline.spark.embedding

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object SparkTest {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("SparkTest")
      .set("spark.submit.deployMode", "client")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    try {
      val spark = SparkSession.builder.config(conf).getOrCreate()
      
      println("✅ Spark Session 创建成功!")
      println(s"Spark 版本: ${spark.version}")
      
      // 简单数据测试
      val data = Seq(1, 2, 3, 4, 5)
      val rdd = spark.sparkContext.parallelize(data)
      val sum = rdd.reduce(_ + _)
      
      println(s"数据测试通过，求和结果: $sum")
      
      spark.stop()
      println("✅ Spark 测试完成!")
      
    } catch {
      case e: Exception => 
        println(s"❌ Spark 测试失败: ${e.getMessage}")
        e.printStackTrace()
    }
  }
}
