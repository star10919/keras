import numpy as np
from icecream import ic

### 데이터 로드하기

x_train_fashion = np.load('./_save/_npy/k55_x_train_fashtion.npy')
x_test_fashion = np.load('./_save/_npy/k55_x_test_fashtion.npy')
y_train_fashion = np.load('./_save/_npy/k55_y_train_fashtion.npy')
y_test_fashion = np.load('./_save/_npy/k55_y_test_fashtion.npy')

ic(x_train_fashion)
ic(x_test_fashion)
ic(y_train_fashion)
ic(y_test_fashion)
ic(x_train_fashion.shape, x_test_fashion.shape, y_train_fashion.shape, y_test_fashion.shape)

'''
 x_train_fashion.shape: (60000, 28, 28)
    x_test_fashion.shape: (10000, 28, 28)
    y_train_fashion.shape: (60000,)
    y_test_fashion.shape: (10000,)
'''

# np.save('./_save/_npy/k55_x_train_fashtion.npy', arr=x_train_fashion)
# np.save('./_save/_npy/k55_x_test_fashtion.npy', arr=x_test_fashion)
# np.save('./_save/_npy/k55_y_train_fashtion.npy', arr=y_train_fashion)
# np.save('./_save/_npy/k55_y_test_fashtion.npy', arr=y_test_fashion)