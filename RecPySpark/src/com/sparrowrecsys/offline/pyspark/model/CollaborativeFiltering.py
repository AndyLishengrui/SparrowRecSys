from pyspark import SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as F

if __name__ == '__main__':
    # 配置Spark
    conf = SparkConf().setAppName('collaborativeFiltering').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    # 文件路径，请更改为你的本地路径
    file_path = 'file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources'
    ratingResourcesPath = file_path + '/webroot/sampledata/ratings.csv'
    
    # 读取评分数据，并将userId、movieId和rating列转换为整数和浮点数类型
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingResourcesPath) \
        .withColumn("userIdInt", F.col("userId").cast(IntegerType())) \
        .withColumn("movieIdInt", F.col("movieId").cast(IntegerType())) \
        .withColumn("ratingFloat", F.col("rating").cast(FloatType()))
    
    # 将数据分为训练集和测试集，比例为80:20
    training, test = ratingSamples.randomSplit((0.8, 0.2))
    
    # 使用ALS在训练数据上构建推荐模型
    # 注意，我们设置冷启动策略为'drop'，以确保不会得到NaN的评估指标
    als = ALS(regParam=0.01, maxIter=5, userCol='userIdInt', itemCol='movieIdInt', ratingCol='ratingFloat',
              coldStartStrategy='drop')
    model = als.fit(training)
    
    # 通过在测试数据上计算RMSE来评估模型
    predictions = model.transform(test)
    model.itemFactors.show(10, truncate=False)  # 显示前10个物品因子
    model.userFactors.show(10, truncate=False)  # 显示前10个用户因子
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol='ratingFloat', metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = {}".format(rmse))  # 打印RMSE
    
    # 为每个用户生成前10个电影推荐
    userRecs = model.recommendForAllUsers(10)
    # 为每个电影生成前10个用户推荐
    movieRecs = model.recommendForAllItems(10)
    # 为指定的一组用户生成前10个电影推荐
    users = ratingSamples.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, 10)
    # 为指定的一组电影生成前10个用户推荐
    movies = ratingSamples.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    
    # 显示推荐结果
    userRecs.show(5, False)
    movieRecs.show(5, False)
    userSubsetRecs.show(5, False)
    movieSubSetRecs.show(5, False)
    
    # 构建参数网格
    paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.01]).build()
    # 创建交叉验证器
    cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    # 使用测试数据进行交叉验证
    cvModel = cv.fit(test)
    # 获取平均评估指标
    avgMetrics = cvModel.avgMetrics
    
    # 停止SparkSession
    spark.stop()