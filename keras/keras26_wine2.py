import numpy as np
import pandas as pd
from icecream import ic
from pandas.core.tools.datetimes import Scalar

# 다중분류
# 모델링하고
# 0.8 이상 완성


# 1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',       # 경로잡기 중요!
                        index_col=None, header=0)    #header=0 첫번째라인

    # * visual studio 기준(파이참이랑 다름)
# ./  :  현재폴더(STUDY)
# ../ :  상위폴더

# print(datasets)   # quality 가  y

    # 아래 3개는 꼭 찍어보기
# ic(datasets.shape)   # (4898, 12)   => x:(4898,11),   y:(4898,) 으로 잘라주기
# ic(datasets.info())
# ic(datasets.describe())




#1 판다스 -> 넘파이 : index와 header가 날아감
#2 x와 y를 분리
#3. sklearn의 onehot??? 사용할것
#3 y의 라벨을 확인 np.unique(y)
#5. y의 shape 확인 (4898,) -> (4898,7)


datasets_np = datasets.to_numpy()   #1 판다스 -> 넘파이
ic(datasets_np)
x = datasets_np[:,0:11]
ic(x)
y = datasets_np[:,[-1]]
ic(y)
ic(x.shape, y.shape)   # x.shape: (4898, 11), y.shape: (4898,)
ic(np.unique(y))   # [3, 4, 5, 6, 7, 8, 9]

# 원핫인코딩
# from tensorflow.keras.utils import to_categorical   # to_categorical 0, 1, 2 없으나 자동 생성
# y = to_categorical(y)
# ic(y)

from sklearn.preprocessing import OneHotEncoder    # 0, 1, 2 자동 채움 안됨
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
onehot = OneHotEncoder()
onehot.fit(y)
y = onehot.transform(y).toarray() 
ic(y.shape)    # (4898, 7)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=9)

# 데이터 전처리(scaler)
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(11,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))


# 3. 컴파일(EalryStopping), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.01, callbacks=[es])


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('accuracy :', results[1])

ic(y_test[-5:-1])
y_predict = model.predict(x_test)
ic(y_predict[-5:-1])


'''
loss : 1.4081991910934448
accuracy : 0.5680271983146667
'''