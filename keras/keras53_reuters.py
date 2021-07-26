from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from icecream import ic

(x_train, y_train),(X_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

ic(x_train[0], type(x_train[0]))

