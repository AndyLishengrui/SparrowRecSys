package com.sparrowrecsys.offline.spark.embedding

import java.io.{BufferedWriter, File, FileWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Row, SparkSession}
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}

object EmbeddingLargeDataset {

  // 处理物品序列数据 - 使用新的大数据集
  def processItemSequence(sparkSession: SparkSession, ratingsPath: String): RDD[Seq[String]] = {
    println(s"正在加载数据集: $ratingsPath")
    
    val ratingSamples = sparkSession.read
      .format("csv")
      .option("header", "true")
      .load(ratingsPath)

    println("数据集Schema:")
    ratingSamples.printSchema()
    
    println(s"总评分数量: ${ratingSamples.count()}")

    // 按时间戳排序的UDF
    val sortUdf: UserDefinedFunction = udf((rows: Seq[Row]) => {
      rows.map { case Row(movieId: String, timestamp: String) => (movieId, timestamp) }
        .sortBy { case (_, timestamp) => timestamp.toLong }
        .map { case (movieId, _) => movieId }
    })

    // 筛选高评分(>=4.0)并生成用户行为序列
    val userSeq = ratingSamples
      .where(col("rating") >= 4.0)  // 提高评分阈值
      .groupBy("userId")
      .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
      .withColumn("movieIdStr", array_join(col("movieIds"), " "))
      .filter(size(col("movieIds")) >= 5)  // 至少5个评分记录

    println("生成的用户序列样例:")
    userSeq.select("userId", "movieIdStr").show(5, truncate = false)
    
    println(s"有效用户数量: ${userSeq.count()}")

    userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
  }

  // 训练Item2Vec模型 - 优化参数
  def trainItem2vecLarge(sparkSession: SparkSession, samples: RDD[Seq[String]], embLength: Int, embOutputFilename: String): Word2VecModel = {
    println("开始训练Item2Vec模型...")
    
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)
      .setWindowSize(10)        // 增大窗口大小
      .setNumIterations(20)     // 增加迭代次数
      .setMinCount(5)           // 最少出现次数

    val model = word2vec.fit(samples)
    
    println(s"训练完成! 词汇表大小: ${model.getVectors.size}")

    // 测试相似性
    if (model.getVectors.contains("1")) {
      val synonyms = model.findSynonyms("1", 10)
      println("电影ID=1的相似电影:")
      for ((synonym, cosineSimilarity) <- synonyms) {
        println(f"  电影ID: $synonym, 相似度: $cosineSimilarity%.4f")
      }
    }

    // 保存embedding
    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))
    
    println(s"保存embedding到: ${file.getAbsolutePath}")
    
    for (movieId <- model.getVectors.keys) {
      bw.write(movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
    }
    bw.close()

    println(s"✅ Embedding已保存到: $embOutputFilename")
    
    model
  }

  // 生成用户嵌入 - 基于大数据集
  def generateUserEmbLarge(sparkSession: SparkSession, ratingsPath: String, word2VecModel: Word2VecModel, embLength: Int, embOutputFilename: String): Unit = {
    println("开始生成用户embedding...")
    
    val ratingSamples = sparkSession.read.format("csv").option("header", "true").load(ratingsPath)
    
    val userEmbeddings = new ArrayBuffer[(String, Array[Float])]()

    // 按用户分组并计算embedding
    ratingSamples
      .filter(col("rating") >= 4.0)
      .collect()
      .groupBy(_.getAs[String]("userId"))
      .foreach { case (userId, userRatings) =>
        var userEmb = new Array[Float](embLength)
        var movieCount = 0
        
        userRatings.foreach { row =>
          val movieId = row.getAs[String]("movieId")
          val movieEmb = word2VecModel.getVectors.get(movieId)
          
          if (movieEmb.isDefined) {
            movieCount += 1
            for (i <- userEmb.indices) {
              userEmb(i) += movieEmb.get(i)
            }
          }
        }
        
        if (movieCount > 0) {
          userEmb = userEmb.map(_ / movieCount)
          userEmbeddings.append((userId, userEmb))
        }
      }

    // 保存用户embedding
    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))

    for ((userId, userEmb) <- userEmbeddings) {
      bw.write(userId + ":" + userEmb.mkString(" ") + "\n")
    }
    bw.close()

    println(s"✅ 用户Embedding已保存: $embOutputFilename, 用户数量: ${userEmbeddings.length}")
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("EmbeddingLargeDataset")
      .set("spark.submit.deployMode", "client")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.sql.adaptive.enabled", "true")
      .set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    try {
      // 使用新的大数据集
      val largRatingsPath = this.getClass.getResource("/webroot/sampledata/ml-25m/ratings.csv").getPath
      val embLength = 50  // 增加embedding维度

      println("🚀 开始使用大数据集训练embedding模型...")
      
      // 处理数据并训练模型
      val samples = processItemSequence(spark, largRatingsPath)
      
      println(s"生成的序列数量: ${samples.count()}")
      
      // 训练Item2Vec
      val model = trainItem2vecLarge(spark, samples, embLength, "item2vecEmb_large.csv")
      
      // 生成用户embedding（可选 - 因为数据量大，可能需要较长时间）
      // generateUserEmbLarge(spark, largRatingsPath, model, embLength, "userEmb_large.csv")
      
      println("🎉 大数据集embedding训练完成!")
      
    } catch {
      case e: Exception =>
        println(s"❌ 训练失败: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}
