import numpy as np

### 시계열 데이터는 직접 x와 y를 나눠줘야 함 (ex) 7/1-4(x) 주식데이터로 7/5(y) 데이터 예측)

a= np.array(range(1, 11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

print(dataset)

x = dataset[:, :4]
y = dataset[:, 4]

print("x :\n", x)
print("y :", y)