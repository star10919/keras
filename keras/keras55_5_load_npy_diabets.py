import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_diabet = np.load('./_save/_npy/k55_x_data_diabet.npy')
y_data_diabet = np.load('./_save/_npy/k55_y_data_diabet.npy')

ic(x_data_diabet)
ic(y_data_diabet)
ic(x_data_diabet.shape, y_data_diabet.shape)        # x_data_diabet.shape: (442, 10), y_data_diabet.shape: (442,)

# np.save('./_save/_npy/k55_x_data_diabet.npy', arr=x_data_diabet)
# np.save('./_save/_npy/k55_y_data_diabet.npy', arr=y_data_diabet)