package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object FeatureEngineering {
  /**
   * One-hot编码示例函数
   * @param samples 电影样本数据框
   */
  def oneHotEncoderExample(samples: DataFrame): Unit = {
    // 将movieId列转换为整数类型
    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))

    // 创建OneHotEncoderEstimator对象
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber")) // 设置输入列
      .setOutputCols(Array("movieIdVector")) // 设置输出列
      .setDropLast(false) // 不丢弃最后一个类别

    // 拟合并转换数据
    val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.printSchema() // 打印Schema
    oneHotEncoderSamples.show(10) // 显示前10条数据
  }

  // 定义UDF将数组转换为稀疏向量
  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) =>
    org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0))
  }

  /**
   * Multi-hot编码示例函数
   * @param samples 电影样本数据框
   */
  def multiHotEncoderExample(samples: DataFrame): Unit = {
    // 分割genres列并展开
    val samplesWithGenre = samples.select(col("movieId"), col("title"), explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
    // 创建StringIndexer对象
    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")

    // 拟合StringIndexer模型
    val stringIndexerModel: StringIndexerModel = genreIndexer.fit(samplesWithGenre)

    // 转换数据并将genreIndex列转换为整数类型
    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))

    // 获取最大索引值
    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1

    // 按movieId分组并收集genreIndexInt列
    val processedSamples = genreIndexSamples
      .groupBy(col("movieId")).agg(collect_list("genreIndexInt").as("genreIndexes"))
      .withColumn("indexSize", typedLit(indexSize))

    // 将genreIndexes列转换为稀疏向量
    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))
    finalSample.printSchema() // 打印Schema
    finalSample.show(10) // 显示前10条数据
  }

  // 定义UDF将Double值转换为密集向量
  val double2vec: UserDefinedFunction = udf { (value: Double) =>
    org.apache.spark.ml.linalg.Vectors.dense(value)
  }

  /**
   * 处理评分样本
   * @param samples 评分样本数据框
   */
  def ratingFeatures(samples: DataFrame): Unit = {
    samples.printSchema() // 打印Schema
    samples.show(10) // 显示前10条数据

    // 计算每部电影的平均评分和评分次数
    val movieFeatures = samples.groupBy(col("movieId"))
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))
      .withColumn("avgRatingVec", double2vec(col("avgRating")))

    movieFeatures.show(10) // 显示前10条电影特征数据

    // 分桶
    val ratingCountDiscretizer = new QuantileDiscretizer()
      .setInputCol("ratingCount")
      .setOutputCol("ratingCountBucket")
      .setNumBuckets(100)

    // 归一化
    val ratingScaler = new MinMaxScaler()
      .setInputCol("avgRatingVec")
      .setOutputCol("scaleAvgRating")

    // 创建Pipeline
    val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    // 拟合并转换数据
    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10) // 显示前10条处理后的电影特征数据
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR) // 设置日志级别为ERROR

    val conf = new SparkConf()
      .setMaster("local") // 设置为本地模式
      .setAppName("featureEngineering") // 设置应用名称为"featureEngineering"
      .set("spark.submit.deployMode", "client") // 设置部署模式为客户端

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    println("Raw Movie Samples:")
    movieSamples.printSchema() // 打印原始电影样本的Schema
    movieSamples.show(10) // 显示前10条原始电影样本数据

    println("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples) // 调用OneHotEncoder示例函数

    println("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples) // 调用MultiHotEncoder示例函数

    println("Numerical features Example:")
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingFeatures(ratingSamples) // 调用处理评分样本函数
  }
}