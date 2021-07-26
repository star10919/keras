from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요','추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글쎄요','별로에요','생각보다 지루해요',
        '연기가 어색해요','재미없어요','너무 재미없다','참 재밌네요','청순이가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)        # 단어종류 28개
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '청순이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences       # 리스트 내 크기가 각각 다를 때 패딩으로 채워줌 / 리스트 내에서는 크기가 달라도 됨
pad_x = pad_sequences(x, padding='pre', maxlen=5)       # pre : 앞/ post : 뒤       # maxlen의 크기보다 길면 maxlen의 크기에 맞춰서 잘리는데 무조건 앞이 잘림!!(한국말은 끝까지 들어라~~뒤가중요하니까)
print(pad_x)
print(pad_x.shape)      # (13, 5)

# 원핫인코딩 하면? (13, 5) -> (13, 5, 27)         # 열(맨 뒤)은 label의 개수