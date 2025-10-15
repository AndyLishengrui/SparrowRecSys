package com.sparrowrecsys.offline.spark.embedding

import java.io.{BufferedWriter, File, FileWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object SimpleEmbeddingTrainer {

  // 简单的数据处理方法 - 只使用RDD
  def processRatingsData(sparkSession: SparkSession, ratingsPath: String): RDD[Seq[String]] = {
    println(s"开始处理数据: $ratingsPath")
    
    // 读取CSV文件作为文本
    val ratingsRDD = sparkSession.sparkContext.textFile(ratingsPath)
    
    // 跳过标题行并解析数据
    val parsedRatings = ratingsRDD
      .filter(line => !line.startsWith("userId"))  // 跳过标题
      .map(line => {
        val parts = line.split(",")
        if (parts.length >= 4) {
          val userId = parts(0)
          val movieId = parts(1) 
          val rating = parts(2).toDouble
          val timestamp = parts(3).toLong
          Some((userId, movieId, rating, timestamp))
        } else {
          None
        }
      })
      .filter(_.isDefined)
      .map(_.get)

    println(s"解析的评分记录数: ${parsedRatings.count()}")

    // 筛选高评分记录(>=4.0)并按用户分组
    val userMovieSequences = parsedRatings
      .filter(_._3 >= 4.0)  // 过滤高评分
      .groupBy(_._1)        // 按用户ID分组
      .map { case (userId, ratings) =>
        // 按时间排序并提取电影ID序列
        val movieSequence = ratings.toSeq
          .sortBy(_._4)  // 按时间戳排序
          .map(_._2)     // 提取电影ID
        (userId, movieSequence)
      }
      .filter(_._2.length >= 5)  // 至少5个评分

    println(s"有效用户数: ${userMovieSequences.count()}")
    
    // 显示一些样例
    userMovieSequences.take(5).foreach { case (userId, movieSeq) =>
      println(s"用户 $userId: ${movieSeq.take(10).mkString(" ")}...")
    }

    // 返回电影序列用于Word2Vec训练
    userMovieSequences.map(_._2)
  }

  // 训练Item2Vec模型
  def trainSimpleItem2Vec(sparkSession: SparkSession, sequences: RDD[Seq[String]], embLength: Int, outputFile: String): Word2VecModel = {
    println("开始训练Item2Vec模型...")
    
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)
      .setWindowSize(10)
      .setNumIterations(10)
      .setMinCount(5)

    val model = word2vec.fit(sequences)
    
    println(s"训练完成! 词汇表大小: ${model.getVectors.size}")

    // 测试相似性
    val movieIds = model.getVectors.keys.toSeq
    if (movieIds.nonEmpty) {
      val testMovieId = movieIds.head
      try {
        val synonyms = model.findSynonyms(testMovieId, 5)
        println(s"电影 $testMovieId 的相似电影:")
        synonyms.foreach { case (movieId, similarity) =>
          println(f"  $movieId: $similarity%.4f")
        }
      } catch {
        case e: Exception => println(s"相似性测试失败: ${e.getMessage}")
      }
    }

    // 保存模型
    saveEmbedding(model, outputFile)
    model
  }

  // 保存embedding到文件
  def saveEmbedding(model: Word2VecModel, filename: String): Unit = {
    try {
      val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
      val file = new File(embFolderPath.getPath + filename)
      val bw = new BufferedWriter(new FileWriter(file))

      println(s"保存embedding到: ${file.getAbsolutePath}")
      
      for (movieId <- model.getVectors.keys) {
        bw.write(movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
      }
      bw.close()
      
      println(s"✅ 成功保存 ${model.getVectors.size} 个电影embedding到 $filename")
    } catch {
      case e: Exception => 
        println(s"❌ 保存embedding失败: ${e.getMessage}")
    }
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("SimpleEmbeddingTrainer")
      .set("spark.submit.deployMode", "client")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    try {
      println("🚀 开始简单embedding训练...")

      // 使用大数据集
      val largRatingsPath = this.getClass.getResource("/webroot/sampledata/ml-25m/ratings.csv").getPath
      val embLength = 50

      // 处理数据
      val sequences = processRatingsData(spark, largRatingsPath)
      
      println(s"生成的序列数: ${sequences.count()}")

      // 训练模型
      val model = trainSimpleItem2Vec(spark, sequences, embLength, "item2vec_large_simple.csv")

      println("🎉 训练完成!")
      
    } catch {
      case e: Exception =>
        println(s"❌ 训练失败: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}
