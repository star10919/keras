import numpy as np
from icecream import ic

### 데이터 로드하기

x_train_mnist = np.load('./_save/_npy/k55_x_train_mnist.npy')
x_test_mnist = np.load('./_save/_npy/k55_x_test_mnist.npy')
y_train_mnist = np.load('./_save/_npy/k55_y_train_mnist.npy')
y_test_mnist = np.load('./_save/_npy/k55_y_test_mnist.npy')

ic(x_train_mnist)
ic(x_test_mnist)
ic(y_train_mnist)
ic(y_test_mnist)
ic(x_train_mnist.shape, x_test_mnist.shape, y_train_mnist.shape, y_test_mnist.shape)

'''
 x_train_mnist.shape: (60000, 28, 28)
    x_test_mnist.shape: (10000, 28, 28)
    y_train_mnist.shape: (60000,)
    y_test_mnist.shape: (10000,)
'''

# (x_train_minst, y_train_mnist), (x_test_minst, y_test_minst) = mnist.load_data()
# np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train_minst)
# np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test_minst)
# np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train_mnist)
# np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test_minst)