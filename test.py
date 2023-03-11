# coding=gbk
import random
from model import MLP
from keras.datasets import mnist
from keras.utils import normalize
import matplotlib.pyplot as plt


#--------------------------------设置路径--------------------------------------
data_path = "D:/科研/python代码/炼丹手册/MLP/dataset/mnist.npz"
save_path = "D:/科研/python代码/炼丹手册/MLP/save_model/mlp.h5"
#-----------------------------------------------------------------------------


#-----------------------------读取训练好的模型----------------------------------
# 定义模型，参数要保持一致
mlp = MLP(
    input_dim = 28*28*1,
    hidden_dim=128,
    output_dim=10,
    )
# 使用build方法构建模型，然后加载weights参数
mlp.build(input_shape=(None, 28, 28))
mlp.load_weights(save_path)
mlp.summary()
#-----------------------------------------------------------------------------


#---------------------------从测试数据集中随机选一张图片--------------------------
_, (testX, test_Y) = mnist.load_data(path=data_path)
idx = random.randint(0, len(testX))
img, label = testX[idx], test_Y[idx]
plt.figure('mnist')
plt.imshow(img, cmap="Greys_r")
plt.show()
#-----------------------------------------------------------------------------


#------------------------------------进行预测----------------------------------
img = normalize(img)
img = img.reshape(1,28,28)
p = mlp.predict(img)
print(f"标签值：{label}\t预测值：{p.argmax(-1)}")
#-----------------------------------------------------------------------------