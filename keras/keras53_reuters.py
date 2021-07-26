from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from icecream import ic

(x_train, y_train),(x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

ic(x_train[0], type(x_train[0]))
ic(y_train[0])

ic(len(x_train[0]), len(x_train[11]))  #  len(x_train[0]): 87, len(x_train[11]): 59    # 길이가 다르니까 앞에서부터 패딩해줘야 함

# ic(x_train[0].shape)        # AttributeError: 'list' object has no attribute 'shape'

ic(x_train.shape, x_test.shape)     # x_train.shape: (8982,), x_test.shape: (2246,)
ic(y_train.shape, y_test.shape)     # y_train.shape: (8982,), y_test.shape: (2246,)

ic(type(x_train))       # type(x_train): <class 'numpy.ndarray'>
