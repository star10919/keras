import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])
# x_pred = [6]

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=1)

loss = model.evaluate(x, y)
print('loss:', loss)

result = model.predict([6])
print('6의 예측값 :', result)


'''
#5. 결과값
loss: 0.3800007700920105
6의 예측값 : [[5.7015915]]
'''

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict, color='red')
plt.show()