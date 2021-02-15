# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    
    def __init__(self, layers_info = [
        'conv', 'relu', 'conv', 'relu', 'pool', # 'conv', 'relu',
        'conv', 'relu', 'conv', 'relu', 'pool',
        'conv', 'relu', 'conv', 'relu', 'pool',
        'affine', 'relu', 'dropout', 'affine', 'dropout', 'softmax'], params = [(1, 28, 28),
        {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
        {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},

        {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
        {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},

        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
        50, 10]):
        # 重みの初期化===========

        # 각 층의 뉴런 하나당 앞층의 뉴런과 연결된 노드 수 저장
        pre_node_nums = np.array([0 for i in range(len(params))])
        conv_params = []
        feature_map_size = params[0][1]
        idx = 0
        for layer in layers_info:
            
            if layer == 'conv' or layer == 'convolutional':
    
                if type(params[idx]) == tuple: ### 'tuple'
                    if type(params[idx+1]) == dict:
                        pre_node_nums[idx] = params[idx][0] * (params[idx+1]['filter_size'] ** 2)
                    else:
                        print(1)
                elif type(params[idx]) == dict:
                    if type(params[idx+1]) == dict:
                        pre_node_nums[idx] = params[idx]['filter_num'] * (params[idx+1]['filter_size'] ** 2)
                    else:
                        print(2)
                    conv_params.append(params[idx])
                else:
                    print(3)
                
                feature_map_size = (( feature_map_size + 2*params[idx+1]['pad'] - params[idx+1]['filter_size'] ) / params[idx+1]['stride']) + 1
                # print(feature_map_size)
                if ( feature_map_size + 2*params[idx+1]['pad'] - params[idx+1]['filter_size'] ) % params[idx+1]['stride'] != 0:
                    print("Error: 합성곱 계층 출력 크기가 정수가 아님")
                idx += 1

            elif layer == 'pool' or layer == 'pooling':
                if feature_map_size % 2 != 0:
                    print("Error: 풀링 계층 출력 크기가 정수가 아님")
                feature_map_size /= 2

            elif layer == 'affine':
                if type(params[idx]) == dict:
                    if type(params[idx+1]) == int:
                        pre_node_nums[idx] = params[idx]['filter_num'] * (feature_map_size**2)
                    else:
                        print(4)
                    conv_params.append(params[idx])
                elif type(params[idx]) == int:
                    if type(params[idx+1]) == int:
                        pre_node_nums[idx] = params[idx]
                    else:
                        print(5)
                else:
                    print(6)

                feature_map_size = pre_node_nums[idx]
                idx += 1
        
        pre_node_nums[idx] = params[-1]
        # pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, 50, 10])
        # print(pre_node_nums)

        # 가중치 초깃값 설정
        layers_info = [layer.lower() for layer in layers_info]
        if 'relu' in layers_info:
            weight_init_scales = np.sqrt(2.0 / pre_node_nums) # He 초깃값
        elif ('sigmoid' in layers_info) or ('tanh' in layers_info):
            weight_init_scales = np.sqrt(1.0 / pre_node_nums) # Xavier 초깃값
        else:
            print("\nError: There is no activation function. (relu or sigmoid or tanh)\n")
            return False
    
        pre_channel_num = params[0][0] ### params[0][1]
        self.params = {} # 매개변수 저장
        self.layers = [] # 레이어 생성
        self.layer_idxs_used_params = [] # 매개변수를 사용하는 레이어의 위치
        idx = 0 
        for layer_idx, layer in enumerate(layers_info):

            if layer == 'conv' or layer == 'convolutional':
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(  conv_params[idx]['filter_num'],
                                                pre_channel_num, conv_params[idx]['filter_size'], conv_params[idx]['filter_size']  )
                self.params['b' + str(idx+1)] = np.zeros(conv_params[idx]['filter_num'])
                pre_channel_num = conv_params[idx]['filter_num']

                self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                                   conv_params[idx]['stride'], conv_params[idx]['pad']))
                self.layer_idxs_used_params.append(layer_idx)
                idx += 1

            elif layer == 'affine':
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn( pre_node_nums[idx], pre_node_nums[idx+1] )
                self.params['b' + str(idx+1)] = np.zeros(pre_node_nums[idx+1])
                self.layers.append(Affine(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)]))
                self.layer_idxs_used_params.append(layer_idx)
                idx += 1

            elif layer == 'relu':
                self.layers.append(Relu())
            elif layer == 'pool' or layer == 'pooling':
                self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
            elif layer == 'dropout' or layer == 'drop':
                self.layers.append(Dropout(0.5))
            elif layer == 'softmax' or layer == 'soft':
                self.last_layer = SoftmaxWithLoss()
            else:
                print(f"\nError: Undefined function.({layer})\n")
                return False

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            # print(layer)
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate(self.layer_idxs_used_params):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate(self.layer_idxs_used_params):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
