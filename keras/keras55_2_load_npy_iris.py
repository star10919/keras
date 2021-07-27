import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_iris = np.load('./_save/_npy/k55_x_data_iris.npy')
y_data_iris = np.load('./_save/_npy/k55_y_data_iris.npy')

ic(x_data_iris)
ic(y_data_iris)
ic(x_data_iris.shape, y_data_iris.shape)        # x_data_iris.shape: (150, 4), y_data_iris.shape: (150,)

# np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
# np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)