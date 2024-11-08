package com.sparrowrecsys.offline.spark.model

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object CollaborativeFiltering {

  def main(args: Array[String]): Unit = {
    // 创建Spark配置
    val conf = new SparkConf()
      .setMaster("local") // 设置为本地模式
      .setAppName("collaborativeFiltering") // 设置应用名称为"collaborativeFiltering"
      .set("spark.submit.deployMode", "client") // 设置部署模式为客户端

    // 创建SparkSession
    val spark = SparkSession.builder.config(conf).getOrCreate()
    // 获取评分数据的路径
    val ratingResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    // 定义UDF将字符串转换为整数
    val toInt = udf[Int, String]( _.toInt)
    // 定义UDF将字符串转换为浮点数
    val toFloat = udf[Double, String]( _.toFloat)
    // 读取评分数据并转换列的数据类型
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcesPath.getPath)
      .withColumn("userIdInt", toInt(col("userId"))) // 将"userId"列转换为整数
      .withColumn("movieIdInt", toInt(col("movieId"))) // 将"movieId"列转换为整数
      .withColumn("ratingFloat", toFloat(col("rating"))) // 将"rating"列转换为浮点数

    // 将数据分为训练集和测试集，比例为80:20
    val Array(training, test) = ratingSamples.randomSplit(Array(0.8, 0.2))

    // 使用ALS在训练数据上构建推荐模型
    val als = new ALS()
      .setMaxIter(5) // 设置最大迭代次数为5
      .setRegParam(0.01) // 设置正则化参数为0.01
      .setUserCol("userIdInt") // 设置用户列为"userIdInt"
      .setItemCol("movieIdInt") // 设置物品列为"movieIdInt"
      .setRatingCol("ratingFloat") // 设置评分列为"ratingFloat"

    // 使用训练数据拟合ALS模型
    val model = als.fit(training)

    // 通过在测试数据上计算RMSE来评估模型
    // 注意，我们设置冷启动策略为'drop'，以确保不会得到NaN的评估指标
    model.setColdStartStrategy("drop") // 设置冷启动策略为'drop'
    val predictions = model.transform(test) // 使用模型对测试数据进行预测

    // 显示前10个物品因子，不截断
    model.itemFactors.show(10, truncate = false)
    // 显示前10个用户因子，不截断
    model.userFactors.show(10, truncate = false)

    // 创建回归评估器
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse") // 设置评估指标为RMSE
      .setLabelCol("ratingFloat") // 设置标签列为"ratingFloat"
      .setPredictionCol("prediction") // 设置预测列为"prediction"
    // 评估预测结果的RMSE
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse") // 打印RMSE

    // 为每个用户生成前10个电影推荐
    val userRecs = model.recommendForAllUsers(10)
    // 为每个电影生成前10个用户推荐
    val movieRecs = model.recommendForAllItems(10)

    // 为指定的一组用户生成前10个电影推荐
    val users = ratingSamples.select(als.getUserCol).distinct().limit(3)
    val userSubsetRecs = model.recommendForUserSubset(users, 10)
    // 为指定的一组电影生成前10个用户推荐
    val movies = ratingSamples.select(als.getItemCol).distinct().limit(3)
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    // 显示推荐结果
    userRecs.show(false)
    movieRecs.show(false)
    userSubsetRecs.show(false)
    movieSubSetRecs.show(false)

    // 构建参数网格
    val paramGrid = new ParamGridBuilder()
      .addGrid(als.regParam, Array(0.01)) // 添加正则化参数的候选值
      .build()

    // 创建交叉验证器
    val cv = new CrossValidator()
      .setEstimator(als) // 设置估计器为ALS
      .setEvaluator(evaluator) // 设置评估器
      .setEstimatorParamMaps(paramGrid) // 设置参数网格
      .setNumFolds(10)  // 设置交叉验证的折数，实际使用中应为3或更多
    // 使用测试数据进行交叉验证
    val cvModel = cv.fit(test)
    // 获取平均评估指标
    val avgMetrics = cvModel.avgMetrics

    // 停止SparkSession
    spark.stop()
  }
}