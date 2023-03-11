# Keras-MLP
基于Keras搭建一个简单的多层感知机（MLP），用mnist数据集对MLP进行训练，完成模型的保存和加载和识别测试。<br /><br />

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1./dataset：保存数据集<br />
2./save_model：保存训练好的模型<br /><br />

Keras有三种构建model的方式：<br />
1.Sequential序列模型<br />
2.函数式API构建模型<br />
3.子类subclass构建模型<br />
不同方式在模型保存和加载方面方法不通用，按照自己习惯选择即可<br /><br />
