import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 請解壓縮data.rar，取得本程式之數據

#---------------讀檔不是很重要---------------#
train_file = 'datasets/train.csv'
test_file = 'datasets/test.csv'

data_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

list_sentences_train = data_df['comment_text'].fillna('NA').values
list_sentences_test  = test_df['comment_text'].fillna('NA').values
#---------------讀檔不是很重要---------------#


# num_words：None或整數，處理的最大單詞數量
# 若被設置為整數，則分詞器將被限制為待處理數據集中最常見的num_words個單詞
tokenizer = Tokenizer(num_words = None , lower = True)
tokenizer.fit_on_texts(list_sentences_train) 

word_index = tokenizer.word_index 
vocab_size = len(word_index)

max_sequence_length = 150
padding = 'post'
sequences = tokenizer.texts_to_sequences(list_sentences_train) # 將list_sentences_train中的詞從文字轉換成數字
train_data = pad_sequences(sequences , 
                           maxlen = max_sequence_length , 
                           padding = padding) # 補零補到長度為max_sequence_length

sequences = tokenizer.texts_to_sequences(list_sentences_test) # 將list_sentences_test中的詞從文字轉換成數字
test_data = pad_sequences(sequences ,
                          maxlen = max_sequence_length ,
                          padding = padding) # 補零補到長度為max_sequence_length

# 讀取glove.6B.300d.txt，
# 直接拿"現成的"embedding_matrix，不一定要訓練embedding_matrix
f = open(os.path.join('datasets' , 'glove.6B.300d.txt') , encoding = 'utf-8') 
lines = f.readlines()
embeddings_index = {}
for line in lines:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1 : ] , dtype = 'float32')
    embeddings_index[word] = coefs


# 產生embedding_matrix
embedding_dim = 300
embedding_matrix = np.zeros((vocab_size , embedding_dim))
embedding_matrix = embedding_matrix.astype(np.float32)
for i , word in enumerate(word_index.keys()):
    if word in embeddings_index.keys():
        # word 有在 embeddings_index.keys()，
        # 才把 embeddings_index[word] 塞到 embedding_matrix[i] 中
        embedding_matrix[i] = embeddings_index[word] 
