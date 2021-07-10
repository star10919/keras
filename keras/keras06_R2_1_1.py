from icecream import ic
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import random
# import tensorflow as tf
# 1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))
# x_train = x[:70]
# y_train = y[:70]
# x_test = x[-30:]
# y_test = y[70:]
# ic(x_train.shape, y_train.shape)  # (70,) (70,)
# ic(x_test.shape, y_test.shape)    # (30,) (30,)
# random.shuffle(x_train)
# random.shuffle(y_train)
# ic(x_train, y_train)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=66)
ic(x_test, y_test)


# 2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic('loss : ', loss)
y_predict = model.predict(x_test) # x_test를 훈련시킨 값으로
# ic('100의 예측값 : ', y_predict)


# R2 결정 계수 : 정확도와 유사한 지표
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)  # y_test와 y_predict값을 통해 결정계수를 계산
ic('R2 스코어 : ', r2)

'''
#5. 결정값
r2스코어 : 'R2 스코어 : ', r2: 0.9999999991403652
'''

