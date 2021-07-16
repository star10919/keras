import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from icecream import ic


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ic(x_train.shape, y_train.shape)   # x_train.shape: (60000, 28, 28), y_train.shape: (60000,)
# ic(x_test.shape, y_test.shape)     # x_test.shape: (10000, 28, 28), y_test.shape: (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)   # reshape : 내용물과 순서는 바뀌지 않음(차원만 바뀜)  -  총 곱셈이 같아야 함.
x_test = x_test.reshape(10000, 28, 28, 1)   # 컨벌루션은 무조건 4차원으로 만들어서 넣어줘야 함.

ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#1-2. 데이터 전처리



# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(30, (2,2), padding='valid'))
model.add(MaxPooling2D())   # 반으로 줄어듬(연산은 안함)
model.add(Conv2D(15, (2,2)))
model.add(Flatten())  # shape자체가 2차원이 됨(Dense써야 되니까-Dense:2차원)(연산은 안함)
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련           metrics['acc]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.024, callbacks=[es])

# 4. 평가, 예측             predict할 필요는 없다
results = model.evaluate(x_test, y_test)
print('binary :', results[0])
print('accruacy :', results[1])


# acc로만 판단해보자.
# 0.98 이상 완성