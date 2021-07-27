import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_wine = np.load('./_save/_npy/k55_x_data_wine.npy')
y_data_wine = np.load('./_save/_npy/k55_y_data_wine.npy')

ic(x_data_wine)
ic(y_data_wine)
ic(x_data_wine.shape, y_data_wine.shape)        # x_data_wine.shape: (178, 13), y_data_wine.shape: (178,)

# np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
# np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)