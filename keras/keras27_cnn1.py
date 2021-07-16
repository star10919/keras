from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D   # 이미지가 2D 이니까


model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(28, 28, 1)))    # 10 : output          # kernel_size=(2,2) : 가로세로2X2로 자르겠다          # 가로28, 세로28, 흑백이라서 1(컬러면 3(RGB))
