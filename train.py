# coding=gbk
from model import MLP
from keras.datasets import mnist
from keras.utils import to_categorical, normalize


#------------------------------设置参数----------------------------------------
classes_num = 10           # 标签类别数
input_dim = 28*28*1        # MLP的输入维度，等于输入图片的H*W*C
hidden_dim = 128           # MLP的hiddent维度
output_dim = 10            # MLP的输出维度，等于输出类别的数量
batch_size = 128           # batch大小
epochs = 5                 # 训练批次
#-----------------------------------------------------------------------------


#------------------------数据集路径、模型保存路径---------------------------------
data_path = "D:/科研/python代码/炼丹手册/MLP/dataset/mnist.npz"
save_path = "D:/科研/python代码/炼丹手册/MLP/save_model/mlp.h5"
#-----------------------------------------------------------------------------


#-------------------加载数据集、归一化、label转one-hot---------------------------
(trainX, trainY), (testX, testY) = mnist.load_data(path=data_path)
trainX = normalize(trainX)
testX = normalize(testX)
trainY = to_categorical(trainY, classes_num)
testY = to_categorical(testY, classes_num)
#-----------------------------------------------------------------------------


#----------------------------搭建模型、编译-------------------------------------
mlp = MLP(input_dim, hidden_dim, output_dim)
mlp.build(input_shape=(None, 28, 28))

mlp.compile(
    optimizer='adam',                       # adam优化器
    loss='categorical_crossentropy',        # 交叉熵损失
    metrics='acc'                           # 准确率指标
    )
mlp.summary()
#-----------------------------------------------------------------------------


#----------------------------训练模型、保存-------------------------------------
mlp.fit(
    x=trainX, 
    y=trainY,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(testX, testY)
    )
mlp.save_weights(save_path)
#-----------------------------------------------------------------------------
