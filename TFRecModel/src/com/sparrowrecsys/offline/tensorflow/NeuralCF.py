import tensorflow as tf

# 训练样本路径，请更改为你的本地路径
training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                     "file:///Users/zhewang/Workspace/SparrowRecSys/src/main"
                                                     "/resources/webroot/sampledata/trainingSamples.csv")
# 测试样本路径，请更改为你的本地路径
test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                 "file:///Users/zhewang/Workspace/SparrowRecSys/src/main"
                                                 "/resources/webroot/sampledata/testSamples.csv")


# 加载样本为 tf 数据集
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,  # 每个批次的样本数量
        label_name='label',  # 标签列的名称
        na_value="0",  # 缺失值填充为0
        num_epochs=1,  # 迭代次数
        ignore_errors=True)  # 忽略错误
    return dataset


# 分割为测试数据集和训练数据集
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

# 电影ID嵌入特征
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)

# 用户ID嵌入特征
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)

# 定义 Keras 模型的输入
inputs = {
    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
}


# neural cf 模型架构一：仅在每个塔中使用嵌入，然后使用 MLP 作为交互层
def neural_cf_model_1(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    # 物品塔
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    # 用户塔
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    # 交互层
    interact_layer = tf.keras.layers.concatenate([item_tower, user_tower])
    # 添加隐藏层
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation='relu')(interact_layer)
    # 输出层
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
    # 创建模型
    neural_cf_model = tf.keras.Model(feature_inputs, output_layer)
    return neural_cf_model


# neural cf 模型架构二：在每个塔中使用嵌入+MLP，然后使用点积层作为输出
def neural_cf_model_2(feature_inputs, item_feature_columns, user_feature_columns, hidden_units):
    # 物品塔
    item_tower = tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(item_tower)

    # 用户塔
    user_tower = tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation='relu')(user_tower)

    # 点积层
    output = tf.keras.layers.Dot(axes=1)([item_tower, user_tower])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    # 创建模型
    neural_cf_model = tf.keras.Model(feature_inputs, output)
    return neural_cf_model


# neural cf 模型架构
model = neural_cf_model_1(inputs, [movie_emb_col], [user_emb_col], [10, 10])

# 编译模型，设置损失函数、优化器和评估指标
model.compile(
    loss='binary_crossentropy',  # 二元交叉熵损失函数
    optimizer='adam',  # Adam优化器
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])  # 评估指标：准确率、ROC AUC、PR AUC

# 训练模型
model.fit(train_dataset, epochs=5)

# 评估模型
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# 打印一些预测结果
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))

# 保存模型
tf.keras.models.save_model(
    model,
    "file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources/webroot/modeldata/neuralcf/002",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)