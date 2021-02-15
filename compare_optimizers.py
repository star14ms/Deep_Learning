# import sys, os
# sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from common.optimizer import *
from modules.plot import *
import time as t

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, dropout_ration=0, use_batchnorm=False)

train_loss = {}
train_accs = {}
test_accs = {}

optimizers = (SGD, Momentum, AdaGrad, Adam, Nesterov, RMSprop)
str_optims = ('SGD', 'Momentum', 'AdaGrad', 'Adam', 'Nesterov', 'RMSprop')

start = t.time()

for optim, str_optim in zip(optimizers, str_optims):
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, dropout_ration=0, use_batchnorm=False)
    optimizer = optim()

    iters_num = 1200 # 10000
    train_size = x_train.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_size / batch_size, 1) # 600
    learning_rate = 0.01
    
    train_loss[str_optim] = []
    train_accs[str_optim] = []
    test_accs[str_optim] = []
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        # batch_mask = np.arange(0, 10)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 勾配 (경사)
        grads = network.gradient(x_batch, t_batch)

        # 更新 (갱신)
        optimizer.update(network.params, grads)
        
        if i % 5 == 0: # 5번 갱신할 때마다 손실 함수 저장
            loss = network.loss(x_batch, t_batch)
            train_loss[str_optim].append(loss)
        else:
            train_loss[str_optim].append(loss)
        
        if i % iter_per_epoch == 0: # 1에폭(600번 갱신) 마다 정확도 저장 
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_accs[str_optim].append(train_acc)
            test_accs[str_optim].append(test_acc)
            print("train acc, test acc | " + str(round(train_acc, 3)) + ", " + str(round(test_acc, 3)))

    print(str_optim, "end")

print("%.4f sec" % (t.time() - start))

# plt.subplot(1, 2, 1)
plot_accuracy_graphs(train_accs, test_accs, str_optims)
plt.show()
# plt.subplot(1, 2, 2)
plot_loss_graphs(train_loss, str_optims)
plt.show()