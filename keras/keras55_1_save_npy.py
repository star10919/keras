from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from icecream import ic

### 넘파이로 데이터 저장 - csv 로 불러오는 거보다 시간 절약 됨

datasets1 = load_iris()

x_data_iris = datasets1.data
y_data_iris = datasets1.target

# ic(type(x_data), type(y_data))      # type(x_data): <class 'numpy.ndarray'>, type(y_data): <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)


datasets2 = load_boston()
x_data_boston = datasets2.data
y_data_boston = datasets2.target
np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)


datasets3 = load_breast_cancer()
x_data_cancer = datasets3.data
y_data_cancer = datasets3.target
np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)


datasets4 = load_diabetes()
x_data_diabet = datasets4.data
y_data_diabet = datasets4.target
np.save('./_save/_npy/k55_x_data_diabet.npy', arr=x_data_diabet)
np.save('./_save/_npy/k55_y_data_diabet.npy', arr=y_data_diabet)


datasets5 = load_wine()
x_data_wine = datasets5.data
y_data_wine = datasets5.target
np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)