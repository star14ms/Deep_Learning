from modules.plot import *
from modules.save_list import *

conv2_f64 = load_list(f"Adagrad_c2_f64_train_loss_list") # 1104s
conv2_f64_2 = load_list(f"Adagrad_c2_f64_train_loss_list2") # 1112s
conv2_f64_3 = load_list(f"Adagrad_c2_f64_train_loss_list3")
conv2_f64_f16p2 = load_list(f"Adagrad_c2_f64_f16p2_train_loss_list") # 1296s

Adam_conv2_f64 = load_list(f"Adam_c2_f64_train_loss_list") # 1313s
RMS_conv2_f64 = load_list(f"RMSprop_c2_f64_train_loss_list") # 1109s
SGD_conv2_f64 = load_list(f"SGD_c2_f64_train_loss_list") # 1265s
Momentum_conv2_f64 = load_list(f"Momentum_c2_f64_train_loss_list") # 1261s

conv3_f64 = load_list(f"Adagrad_c3_f64_train_loss_list") 
conv4_f64 = load_list(f"Adagrad_c4_f64_train_loss_list")

conv2_f128 = load_list(f"Adagrad_c2_f128_train_loss_list") # 1237s
conv2_f128_2 = load_list(f"Adagrad_c2_f128_train_loss_list2") # 1896s

# conv2_f128_p2 = load_list(f"Adagrad_c2_f128_p2_train_loss_list") # 2000s
# conv2_f64_2 = load_list(f"Adagrad222_c2_f64_train_loss_list")

list = {'Adagrad': conv2_f64, 'conv3_f64': conv3_f64, 'conv4_f64': conv4_f64, 
        'conv2_f64_f16p2': conv2_f64_f16p2, 'conv2_f64_2': conv2_f64_2, 'conv2_f64_3': conv2_f64_3,
        'conv2_f128': conv2_f128, 'conv2_f128_2': conv2_f128_2, 'conv2_f128_p2': conv2_f128, 
        'SGD': SGD_conv2_f64, 'Momentum': Momentum_conv2_f64, 'Adam': Adam_conv2_f64, 
        'RMSprop': RMS_conv2_f64}

plot_loss_graphs(list, ('SGD', 'Momentum', 'Adagrad', 'Adam', 'RMSprop'))
plt.show()

# ('conv2_f64', 'conv3_f64', 'conv4_f64', 'conv2_f128')