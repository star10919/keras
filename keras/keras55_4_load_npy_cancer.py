import numpy as np
from icecream import ic

### 데이터 로드하기

x_data_cancer = np.load('./_save/_npy/k55_x_data_cancer.npy')
y_data_cancer = np.load('./_save/_npy/k55_y_data_cancer.npy')

ic(x_data_cancer)
ic(y_data_cancer)
ic(x_data_cancer.shape, y_data_cancer.shape)        # x_data_cancer.shape: (569, 30), y_data_cancer.shape: (569,)

# np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
# np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)