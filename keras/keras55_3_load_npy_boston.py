import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_boston = np.load('./_save/_npy/k55_x_data_boston.npy')
y_data_boston = np.load('./_save/_npy/k55_y_data_boston.npy')

ic(x_data_boston)
ic(y_data_boston)
ic(x_data_boston.shape, y_data_boston.shape)        # x_data_boston.shape: (506, 13), y_data_boston.shape: (506,)

# np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
# np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)