from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

# 将数组转换为稀疏向量的函数
def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()  # 对类型索引进行排序
    fill_list = [1.0 for _ in range(len(genreIndexes))]  # 创建一个填充值为1.0的列表
    return Vectors.sparse(indexSize, genreIndexes, fill_list)  # 返回稀疏向量

# MultiHotEncoder 示例函数
def multiHotEncoderExample(movieSamples):
    # 分割 genres 列并展开
    samplesWithGenre = movieSamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    # 创建 StringIndexer 对象
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    # 拟合 StringIndexer 模型
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    # 转换数据并将 genreIndex 列转换为整数类型
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndexInt",
                                                                                  F.col("genreIndex").cast(IntegerType()))
    # 获取最大索引值
    indexSize = genreIndexSamples.agg(max(F.col("genreIndexInt"))).head()[0] + 1
    # 按 movieId 分组并收集 genreIndexInt 列
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(indexSize))
    # 将 genreIndexes 列转换为稀疏向量
    finalSample = processedSamples.withColumn("vector",
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))
    # 打印 Schema
    finalSample.printSchema()
    # 显示前10条数据
    finalSample.show(10)

# OneHotEncoder 示例函数
def oneHotEncoderExample(movieSamples):
    # 将 movieId 列转换为整数类型
    samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))
    # 创建 StringIndexer 对象
    indexer = StringIndexer(inputCol="movieIdNumber", outputCol="movieIdIndex")
    # 创建 OneHotEncoder 对象
    encoder = OneHotEncoder(inputCol="movieIdIndex", outputCol="movieIdVector")
    # 拟合并转换数据
    indexedSamples = indexer.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples = encoder.fit(indexedSamples).transform(indexedSamples)
    # 打印 Schema
    oneHotEncoderSamples.printSchema()
    # 显示前10条数据
    oneHotEncoderSamples.show(10)

# 处理评分样本的函数
def ratingFeatures(ratingSamples):
    # 打印 Schema
    ratingSamples.printSchema()
    # 显示前10条数据
    ratingSamples.show()
    # 计算每部电影的平均评分和评分次数
    movieFeatures = ratingSamples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),
                                                         F.avg("rating").alias("avgRating"),
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))
    # 显示前10条电影特征数据
    movieFeatures.show(10)
    # 分桶
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")
    # 归一化
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    # 创建 Pipeline
    pipelineStage = [ratingCountDiscretizer, ratingScaler]
    featurePipeline = Pipeline(stages=pipelineStage)
    # 拟合并转换数据
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    # 显示前10条处理后的电影特征数据
    movieProcessedFeatures.show(10)

if __name__ == '__main__':
    # 配置 Spark
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # 更改为你自己的文件路径
    file_path = 'file:///d:/teaching/2024年/2024年学科实践3/SparrowRecSys/src/main/resources'
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"
    # 读取电影数据
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    print("Raw Movie Samples:")
    # 显示前10条原始电影数据
    movieSamples.show(10)
    # 打印原始电影数据的 Schema
    movieSamples.printSchema()
    print("OneHotEncoder Example:")
    # 调用 OneHotEncoder 示例函数
    oneHotEncoderExample(movieSamples)
    print("MultiHotEncoder Example:")
    # 调用 MultiHotEncoder 示例函数
    multiHotEncoderExample(movieSamples)
    print("Numerical features Example:")
    # 读取评分数据
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    # 调用处理评分样本的函数
    ratingFeatures(ratingSamples)