import numpy as np
import matplotlib.pyplot as plt
from .util import smooth_curve
# %matplotlib inline

def plot_accuracy_graphs(train_accs, test_accs, str_optims, colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_accs['SGD']))
    for str_optim, color in zip(str_optims, colors):
        plt.plot(x, train_accs[str_optim], f"-{color}", label=str_optim+'train acc')
        plt.plot(x, test_accs[str_optim], f"--{color}", label=str_optim+'test acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right') # 그래프 이름 표시

def plot_loss_graphs(train_loss, str_optims, colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')):
    x = np.arange(len(train_loss[str_optims[0]]))
    for str_optim, color in zip(str_optims, colors):
        plt.plot(x, smooth_curve(train_loss[str_optim]), f"-{color}", label=str_optim)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)
    plt.legend(loc='upper right')

def plot_accuracy_graph(train_accs, test_accs):
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_accs))
    plt.plot(x, train_accs, f"-", label='train acc')
    plt.plot(x, test_accs, f"--", label='test acc')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right') # 그래프 이름 표시

def plot_loss_graph(train_loss):
    x = np.arange(len(train_loss))
    try:
        plt.plot(x, smooth_curve(train_loss), f"-", label="loss")
    except:
        plt.plot(x, train_loss, f"-", label="loss")
        print("Error: smooth_curve() error")

    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)
    plt.legend(loc='upper right')
