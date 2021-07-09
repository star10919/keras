from sklearn.datasets import load_boston
from tensorflow.keras.models

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape)   # (506, 13)   input_dim=13
print(y.shape)   # (506,)      output = 1(벡터가 1개니까)

print(datasets.feature_names)
print(datasets.DESCR)   # DESCR : 묘사하다

# train_test_split(70%) 사용,       loss 값, r2(0.7정도는 넘기기) 값 출력



#2. 모델


#3. 컴파일, 훈련


#4. 평가, 예측


'''
#5. 결과값

'''

