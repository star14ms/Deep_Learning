# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from modules.modules import Affine, Relu, SoftmaxWithLoss, Dropout, BatchNormalization
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, dropout_ration=0, use_batchnorm=False):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = np.sqrt(2.0 / input_size) * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(2.0 / hidden_size) * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        self.use_batchnorm = use_batchnorm

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        
        if self.use_batchnorm:
                self.params['gamma'] = np.ones(hidden_size)
                self.params['beta'] = np.zeros(hidden_size)
                self.layers['BatchNorm'] = BatchNormalization(self.params['gamma'], self.params['beta'])
        
        self.layers['Relu1'] = Relu()
        if 1 >= dropout_ration > 0:
                self.layers['Dropout1'] = Dropout(dropout_ration)
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        if self.use_batchnorm:
                grads['gamma'] = numerical_gradient(loss_W, self.params['gamma'])
                grads['beta'] = numerical_gradient(loss_W, self.params['beta'])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        if self.use_batchnorm:
            grads['gamma'] = self.layers['BatchNorm'].dgamma
            grads['beta'] = self.layers['BatchNorm'].dbeta
        
        return grads
