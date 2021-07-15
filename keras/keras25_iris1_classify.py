import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 다중 분류 (0,1,2를 명목척도로 바꿔줘야 함 <- 원핫인코딩 사용(to_categorical))

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

# 원핫인코딩 One-Hot-Encoding
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

# [0, 1, 2, 1] ->
# [[1, 0, 0]
#  [0, 1, 0]
#  [0, 0, 1]
#  [0, 1, 0]]   (4,) -> (4, 3) (데이터 수, 데이터 종류(0,1,2))

from tensorflow.keras.utils import to_categorical    # 원핫인코딩
y = to_categorical(y)
ic(y[:5])
ic(y.shape)   # (150, 3)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
model = Sequential()
model.add(Dense(120, input_shape=(4,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))  # softmax : 다중분류   # 0,1,2  3개라서 3개로 나와야 함(150, 3)


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min')

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[es])


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)   # 무조건 낮을수록 좋다
print('loss :', results[0])
print('accuracy :', results[1])


# 그래프 그리기
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'gulim'
plt.plot(hist.history['loss'])

plt.title('아이리스')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend('아이리스 loss')
plt.show()