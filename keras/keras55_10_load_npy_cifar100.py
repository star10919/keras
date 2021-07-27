import numpy as np
from icecream import ic

### 데이터 로드하기

x_train_cifar100 = np.load('./_save/_npy/k55_x_train_cifar100.npy')
x_test_cifar100 = np.load('./_save/_npy/k55_x_test_cifar100.npy')
y_train_cifar100 = np.load('./_save/_npy/k55_y_train_cifar100.npy')
y_test_cifar100 = np.load('./_save/_npy/k55_y_test_cifar100.npy')

ic(x_train_cifar100)
ic(x_test_cifar100)
ic(y_train_cifar100)
ic(y_test_cifar100)
ic(x_train_cifar100.shape, x_test_cifar100.shape, y_train_cifar100.shape, y_test_cifar100.shape)

'''
 x_train_cifar100.shape: (50000, 32, 32, 3)
    x_test_cifar100.shape: (10000, 32, 32, 3)
    y_train_cifar100.shape: (50000, 1)
    y_test_cifar100.shape: (10000, 1)
'''

# np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train_cifar100)
# np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test_cifar100)
# np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train_cifar100)
# np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test_cifar100)