from tensorflow.keras.models import Sequential  # Sequential : 순차적으로 내려가는 모델
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5]) 

#2. 모델구성(딥러닝 구현)
model = Sequential()
model.add(Dense(3, input_dim=1))     # 앞이 output, 뒤가 input
model.add(Dense(10))   # hidden layer #(node개수) # input 적지 않음(위의 output이 아래의 input이 되니까)
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))   # output layer

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([6])
print('6의 예측값 :', result)


'''
#5. 결과값

'''