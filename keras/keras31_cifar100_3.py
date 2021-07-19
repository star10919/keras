from tensorflow.keras.datasets import cifar100
import numpy as np
from icecream import ic

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)
ic(np.max(x_train), np.max(x_test))  # 255

# 1-2. x 데이터 전처리 - scaler:2차원에서만 가능
x_train = x_train.reshape(50000, 32 * 32 * 3)    #(50000, 3072)
x_test = x_test.reshape(10000, 32 * 32 * 3)      #(10000, 3072)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)   # fit_transform 은  x_train 에서만 사용한다!!!!!!!!!!!!
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)    # 4차원으로 다시 reshape(Conv2d 사용해야 되니까)
x_test = x_test.reshape(10000, 32, 32, 3)

# 1-3. y 데이터 전처리
ic(np.unique(y_train))   # 100개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder    # sklearn으로 되어 있는 애들은 모두 2차원으로 해줘야 함
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()   # (50000, 100)
y_test = one.transform(y_test).toarray()     # (10000, 100)
# ic(y_train.shape, y_test.shape)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(MaxPool2D())                                         
model.add(Conv2D(128, (4, 4), activation='relu'))                   
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(Flatten())                                              
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()


# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=400, validation_split=0.0025, callbacks=[es])
end_time = time.time() - start_time


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print("=============================================")
print("걸린 시간 :", end_time)
print('category :', results[0])
print('accuracy :', results[1])


'''
걸린 시간 : 50.16567611694336
category : 2.7187631130218506
accuracy : 0.4185999929904938
'''