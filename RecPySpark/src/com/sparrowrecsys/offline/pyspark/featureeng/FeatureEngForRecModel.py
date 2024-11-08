from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.functions import *
from pyspark.sql.types import *
from collections import defaultdict
from pyspark.sql import functions as F

NUMBER_PRECISION = 2  # 数字精度

def addSampleLabel(ratingSamples):
    # 显示前5条评分数据
    ratingSamples.show(5, truncate=False)
    # 打印评分数据的Schema
    ratingSamples.printSchema()
    # 计算样本数量
    sampleCount = ratingSamples.count()
    # 计算每个评分的数量和百分比
    ratingSamples.groupBy('rating').count().orderBy('rating').withColumn('percentage',
                                                                         F.col('count') / sampleCount).show()
    # 添加标签列，评分>=3.5为1，否则为0
    ratingSamples = ratingSamples.withColumn('label', when(F.col('rating') >= 3.5, 1).otherwise(0))
    return ratingSamples

def extractReleaseYearUdf(title):
    # 提取上映年份
    if not title or len(title.strip()) < 6:
        return 1990
    else:
        yearStr = title.strip()[-5:-1]
    return int(yearStr)

def addMovieFeatures(movieSamples, ratingSamplesWithLabel):
    # 添加电影基本特征
    samplesWithMovies1 = ratingSamplesWithLabel.join(movieSamples, on=['movieId'], how='left')
    # 添加上映年份和标题
    samplesWithMovies2 = samplesWithMovies1.withColumn('releaseYear',
                                                       udf(extractReleaseYearUdf, IntegerType())('title')) \
        .withColumn('title', udf(lambda x: x.strip()[:-6].strip(), StringType())('title')) \
        .drop('title')
    # 分割类型
    samplesWithMovies3 = samplesWithMovies2.withColumn('movieGenre1', split(F.col('genres'), "\\|")[0]) \
        .withColumn('movieGenre2', split(F.col('genres'), "\\|")[1]) \
        .withColumn('movieGenre3', split(F.col('genres'), "\\|")[2])
    # 添加评分特征
    movieRatingFeatures = samplesWithMovies3.groupBy('movieId').agg(F.count(F.lit(1)).alias('movieRatingCount'),
                                                                    format_number(F.avg(F.col('rating')),
                                                                                  NUMBER_PRECISION).alias(
                                                                        'movieAvgRating'),
                                                                    F.stddev(F.col('rating')).alias(
                                                                        'movieRatingStddev')).fillna(0) \
        .withColumn('movieRatingStddev', format_number(F.col('movieRatingStddev'), NUMBER_PRECISION))
    # 连接电影评分特征
    samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, on=['movieId'], how='left')
    samplesWithMovies4.printSchema()  # 打印Schema
    samplesWithMovies4.show(5, truncate=False)  # 显示前5条数据
    return samplesWithMovies4

def extractGenres(genres_list):
    '''
    传入一个格式如["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]的列表
    按每个类型计数，返回按计数降序排列的类型列表
    例如：
    输入: ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    返回: ['Thriller', 'Action', 'Sci-Fi', 'Horror', 'Adventure', 'Crime']
    '''
    genres_dict = defaultdict(int)
    for genres in genres_list:
        for genre in genres.split('|'):
            genres_dict[genre] += 1
    sortedGenres = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sortedGenres]

def addUserFeatures(samplesWithMovieFeatures):
    extractGenresUdf = udf(extractGenres, ArrayType(StringType()))
    samplesWithUserFeatures = samplesWithMovieFeatures \
        .withColumn('userPositiveHistory',
                    F.collect_list(when(F.col('label') == 1, F.col('movieId')).otherwise(F.lit(None))).over(
                        sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userPositiveHistory", reverse(F.col("userPositiveHistory"))) \
        .withColumn('userRatedMovie1', F.col('userPositiveHistory')[0]) \
        .withColumn('userRatedMovie2', F.col('userPositiveHistory')[1]) \
        .withColumn('userRatedMovie3', F.col('userPositiveHistory')[2]) \
        .withColumn('userRatedMovie4', F.col('userPositiveHistory')[3]) \
        .withColumn('userRatedMovie5', F.col('userPositiveHistory')[4]) \
        .withColumn('userRatingCount',
                    F.count(F.lit(1)).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn('userAvgReleaseYear', F.avg(F.col('releaseYear')).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)).cast(IntegerType())) \
        .withColumn('userReleaseYearStddev', F.stddev(F.col("releaseYear")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userAvgRating", format_number(
        F.avg(F.col("rating")).over(sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)),
        NUMBER_PRECISION)) \
        .withColumn("userRatingStddev", F.stddev(F.col("rating")).over(
        sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1))) \
        .withColumn("userGenres", extractGenresUdf(
        F.collect_list(when(F.col('label') == 1, F.col('genres')).otherwise(F.lit(None))).over(
            sql.Window.partitionBy('userId').orderBy('timestamp').rowsBetween(-100, -1)))) \
        .withColumn("userRatingStddev", format_number(F.col("userRatingStddev"), NUMBER_PRECISION)) \
        .withColumn("userReleaseYearStddev", format_number(F.col("userReleaseYearStddev"), NUMBER_PRECISION)) \
        .withColumn("userGenre1", F.col("userGenres")[0]) \
        .withColumn("userGenre2", F.col("userGenres")[1]) \
        .withColumn("userGenre3", F.col("userGenres")[2]) \
        .withColumn("userGenre4", F.col("userGenres")[3]) \
        .withColumn("userGenre5", F.col("userGenres")[4]) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)
    samplesWithUserFeatures.printSchema()  # 打印Schema
    samplesWithUserFeatures.show(10)  # 显示前10条数据
    samplesWithUserFeatures.filter(samplesWithMovieFeatures['userId'] == 1).orderBy(F.col('timestamp').asc()).show(
        truncate=False)  # 显示用户ID为1的所有数据，按时间戳升序排列
    return samplesWithUserFeatures

def splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path):
    # 抽样10%的数据
    smallSamples = samplesWithUserFeatures.sample(0.1)
    # 按8:2分割训练集和测试集
    training, test = smallSamples.randomSplit((0.8, 0.2))
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    # 保存训练集
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    # 保存测试集
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)

def splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path):
    # 抽样10%的数据，并将时间戳转换为长整型
    smallSamples = samplesWithUserFeatures.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))
    # 计算80%分位数的时间戳
    quantile = smallSamples.stat.approxQuantile("timestampLong", [0.8], 0.05)
    splitTimestamp = quantile[0]
    # 按时间戳分割训练集和测试集
    training = smallSamples.where(F.col("timestampLong") <= splitTimestamp).drop("timestampLong")
    test = smallSamples.where(F.col("timestampLong") > splitTimestamp).drop("timestampLong")
    trainingSavePath = file_path + '/trainingSamples'
    testSavePath = file_path + '/testSamples'
    # 保存训练集
    training.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(trainingSavePath)
    # 保存测试集
    test.repartition(1).write.option("header", "true").mode('overwrite') \
        .csv(testSavePath)

if __name__ == '__main__':
    # 配置Spark
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    file_path = 'file:///home/hadoop/SparrowRecSys/src/main/resources'
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"
    # 读取电影数据
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)
    # 读取评分数据
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)
    # 添加标签
    ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    ratingSamplesWithLabel.show(10, truncate=False)  # 显示前10条带标签的评分数据
    # 添加电影特征
    samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    # 添加用户特征
    samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
    # 保存样本为CSV格式
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, file_path + "/webroot/sampledata")
    # 按时间戳分割并保存样本
    # splitAndSaveTrainingTestSamplesByTimeStamp(samplesWithUserFeatures, file_path + "/webroot/sampledata")