# coding=gbk
from model import MLP
from keras.datasets import mnist
from keras.utils import to_categorical, normalize


#------------------------------���ò���----------------------------------------
classes_num = 10           # ��ǩ�����
input_dim = 28*28*1        # MLP������ά�ȣ���������ͼƬ��H*W*C
hidden_dim = 128           # MLP��hiddentά��
output_dim = 10            # MLP�����ά�ȣ����������������
batch_size = 128           # batch��С
epochs = 5                 # ѵ������
#-----------------------------------------------------------------------------


#------------------------���ݼ�·����ģ�ͱ���·��---------------------------------
data_path = "D:/����/python����/�����ֲ�/MLP/dataset/mnist.npz"
save_path = "D:/����/python����/�����ֲ�/MLP/save_model/mlp.h5"
#-----------------------------------------------------------------------------


#-------------------�������ݼ�����һ����labelתone-hot---------------------------
(trainX, trainY), (testX, testY) = mnist.load_data(path=data_path)
trainX = normalize(trainX)
testX = normalize(testX)
trainY = to_categorical(trainY, classes_num)
testY = to_categorical(testY, classes_num)
#-----------------------------------------------------------------------------


#----------------------------�ģ�͡�����-------------------------------------
mlp = MLP(input_dim, hidden_dim, output_dim)
mlp.build(input_shape=(None, 28, 28))

mlp.compile(
    optimizer='adam',                       # adam�Ż���
    loss='categorical_crossentropy',        # ��������ʧ
    metrics='acc'                           # ׼ȷ��ָ��
    )
mlp.summary()
#-----------------------------------------------------------------------------


#----------------------------ѵ��ģ�͡�����-------------------------------------
mlp.fit(
    x=trainX, 
    y=trainY,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(testX, testY)
    )
mlp.save_weights(save_path)
#-----------------------------------------------------------------------------
