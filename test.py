# coding=gbk
import random
from model import MLP
from keras.datasets import mnist
from keras.utils import normalize
import matplotlib.pyplot as plt


#--------------------------------����·��--------------------------------------
data_path = "D:/����/python����/�����ֲ�/MLP/dataset/mnist.npz"
save_path = "D:/����/python����/�����ֲ�/MLP/save_model/mlp.h5"
#-----------------------------------------------------------------------------


#-----------------------------��ȡѵ���õ�ģ��----------------------------------
# ����ģ�ͣ�����Ҫ����һ��
mlp = MLP(
    input_dim = 28*28*1,
    hidden_dim=128,
    output_dim=10,
    )
# ʹ��build��������ģ�ͣ�Ȼ�����weights����
mlp.build(input_shape=(None, 28, 28))
mlp.load_weights(save_path)
mlp.summary()
#-----------------------------------------------------------------------------


#---------------------------�Ӳ������ݼ������ѡһ��ͼƬ--------------------------
_, (testX, test_Y) = mnist.load_data(path=data_path)
idx = random.randint(0, len(testX))
img, label = testX[idx], test_Y[idx]
plt.figure('mnist')
plt.imshow(img, cmap="Greys_r")
plt.show()
#-----------------------------------------------------------------------------


#------------------------------------����Ԥ��----------------------------------
img = normalize(img)
img = img.reshape(1,28,28)
p = mlp.predict(img)
print(f"��ǩֵ��{label}\tԤ��ֵ��{p.argmax(-1)}")
#-----------------------------------------------------------------------------