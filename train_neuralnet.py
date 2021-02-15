# coding: utf-8 '###': 실수한 부분

# import sys, os
# sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from modules.plot import *
import time as t
import datetime as dt
from modules.save_list import *
from selenium import webdriver

# mnist 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
network = DeepConvNet()

# 네트워크로 주고받는 데이터의 비트 수 줄이기 64bit -> 16bit
x_train = x_train.astype(np.float16)
t_train = t_train.astype(np.float16)
x_test = x_test.astype(np.float16)
t_test = t_test.astype(np.float16) ### x_test =
for param in network.params.values():
    param[...] = param.astype(np.float16)

start = t.time()

# 학습
optimizer='Adagrad'# 매개변수 최적화 방법 선택 (SGD, Momentum, AdaGrad, Adam, Nesterov, RMSprop) 
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=1, mini_batch_size=100,
                  optimizer=optimizer, optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000, verbose=True)
trainer.train() # 학습 시작

# 손실 함수, 정확도 변화 리스트 저장
file_name = f'{optimizer}_c2_f64_norm'
save_list_to_file(trainer.train_loss_list, f"{file_name}_train_loss_list")
# save_list_to_file(trainer.train_acc_list, f"{file_name}_train_acc_list")
# save_list_to_file(trainer.test_acc_list, f"{file_name}_test_acc_list")

# 학습 소요 시간 출력
print("%.4f sec" % (t.time() - start))

# 학습 끝나면 알람 울리기
now = dt.datetime.today()  
if int(now.strftime('%S')) < 52:
    alarm_time = now + dt.timedelta(minutes=1)
else:
    alarm_time = now + dt.timedelta(minutes=2)
alarm_time = alarm_time.strftime('%X')
driver = webdriver.Chrome(r'C:\Users\danal\OneDrive\바탕 화면\programing\chromedriver.exe')
driver.get(f'https://vclock.kr/#time={alarm_time}&title=%EC%95%8C%EB%9E%8C&sound=musicbox&loop=1')
driver.find_element_by_xpath('//*[@id="pnl-main"]').click()

# 그래프 출력하기
plot_loss_graph(trainer.train_loss_list)
plt.show()
# plot_accuracy_graph(trainer.train_acc_list, trainer.test_acc_list)
# plt.show()
