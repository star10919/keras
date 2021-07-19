import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from icecream import ic

### dnn(Dense, 2차원)이 cnn(Convolution, 4차원) 보다 연산량이 훨씬 적음
# DNN 구해서 CNN 과 비교
# DNN + GlobalAveragePooling 구해서 CNN 비교

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ic(x_train.shape, y_train.shape)   # x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
# ic(x_test.shape, y_test.shape)     # x_test.shape: (10000, 28, 28), y_test.shape: (10000,)

x_train = x_train.reshape(60000, 28 * 28)   #  convolution을 dense(2차원) 으로 만들 수 있음
x_test = x_test.reshape(10000, 28 * 28)

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# #1-2. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)
# ic(y_train.shape, y_test.shape)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D

model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(4,4), padding='same', input_shape=(28, 28, 1)))
model.add(Dense(100, input_shape=(28 * 28,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()


# 3. 컴파일, 훈련           metrics['acc]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.0012, callbacks=[es])

# 4. 평가, 예측             predict할 필요는 없다
results = model.evaluate(x_test, y_test)
print('category :', results[0])
print('accruacy :', results[1])


# acc로만 판단해보자.
# 0.98 이상 완성


'''
category : 0.04307371377944946
accruacy : 0.9900000095367432
'''