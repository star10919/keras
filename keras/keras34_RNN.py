import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from icecream import ic

### RNN - 3차원

# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

ic(x.shape, y.shape)   #  x.shape: (4, 3), y.shape: (4,)

x = x.reshape(4, 3, 1)   # (batch_size , timesteps, feature)   *feature : 몇 개씩 자르는지(가장 작은 연산의 단위)


# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))   # units - output node 개수  # input_shape 에는 항상 데이터의 개수(맨 앞) 무시하고 쓰기
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# 3. 컴파일(ES), 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_input)
print(results)
