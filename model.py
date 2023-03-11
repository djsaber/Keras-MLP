# coding=gbk
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model



class MLP(Model):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        
        self.flatten = Flatten()
        self.den_1 = Dense(input_dim, activation='relu')
        self.den_2 = Dense(hidden_dim, activation='relu')
        self.den_3 = Dense(output_dim, activation='softmax')
        self.drop_1 = Dropout(rate = 0.5)
        self.drop_2 = Dropout(rate = 0.5)

        # ����Input��ά�ȣ�ʹKeras����ܹ��Զ��ƶϸ���ά����Ϣ
        self.input_layer = Input((28,28,))
        self.out = self.call(self.input_layer)


    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.den_1(x)
        x = self.drop_1(x)
        x = self.den_2(x)
        x = self.drop_2(x)
        x = self.den_3(x)
        return x
