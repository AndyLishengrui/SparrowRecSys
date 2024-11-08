package com.sparrowrecsys.offline.spark.evaluate

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

class Evaluator {
  // 定义评估方法，输入为预测结果的数据框
  def evaluate(predictions: DataFrame): Unit = {

    // 导入隐式转换
    import predictions.sparkSession.implicits._

    // 提取标签和预测概率，并转换为RDD
    val scoreAndLabels = predictions.select("label", "probability").map { row =>
      // 获取概率向量的第二个元素（即正类的概率）和标签
      (row.apply(1).asInstanceOf[DenseVector](1), row.getAs[Int]("label").toDouble)
    }

    // 创建二分类评估指标对象
    val metrics = new BinaryClassificationMetrics(scoreAndLabels.rdd)

    // 打印PR曲线下面积
    println("AUC under PR = " + metrics.areaUnderPR())
    // 打印ROC曲线下面积
    println("AUC under ROC = " + metrics.areaUnderROC())
  }
}