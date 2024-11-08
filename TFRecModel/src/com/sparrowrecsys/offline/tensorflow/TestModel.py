import tensorflow as tf  # 导入TensorFlow库
mnist = tf.keras.datasets.mnist  # 导入MNIST数据集

# 加载MNIST数据集，分为训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 将图像像素值归一化到0到1之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 创建一个顺序模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将输入的28x28的图像展平成一维向量
    tf.keras.layers.Dense(128, activation='relu'),  # 全连接层，128个神经元，ReLU激活函数
    tf.keras.layers.Dropout(0.2),  # Dropout层，丢弃20%的神经元，防止过拟合
    tf.keras.layers.Dense(10, activation='softmax')  # 输出层，10个神经元，Softmax激活函数，用于分类
])

# 编译模型，设置优化器、损失函数和评估指标
model.compile(optimizer='adam',  # Adam优化器
              loss='sparse_categorical_crossentropy',  # 稀疏分类交叉熵损失函数
              metrics=['accuracy'])  # 评估指标：准确率

# 训练模型，设置训练数据、标签和训练轮数
model.fit(x_train, y_train, epochs=5)

# 评估模型，使用测试数据和标签
model.evaluate(x_test, y_test, verbose=2)