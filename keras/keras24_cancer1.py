import numpy as np
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split



# 1. 데이터
datasets = load_breast_cancer()
# ic(datasets.DESCR)
# ic(datasets.feature_names)

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)   #(569, 30), (569,)
# ic(y[:40])  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# ic(np.unique(y))   # [0, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)


# 2. 모델
model = Sequential()
model.add(Dense(120, input_shape=(30,), activation='relu'))
model.add(Dense(64), activation='relu')
model.add(Dense(64), activation='relu')
model.add(Dense(32), activation='relu')
model.add(Dense(32), activation='relu')
model.add(Dense(16), activation='relu')
model.add(Dense(4), activation='relu')
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

