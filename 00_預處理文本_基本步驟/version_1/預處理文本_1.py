import zipfile
import numpy as np
import tensorflow as tf

#---------------讀檔不是很重要---------------#
filename = 'text8.zip'

print (filename)
# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of vocabulary."""
    f = zipfile.ZipFile(filename)
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))
#---------------讀檔不是很重要---------------#


# 1.決定只考慮文本中最常出現的50000個詞，其餘不常出現的字都會變成'UNK'
vocabulary_size = 50000 
count = {} # count內含有單字與該單字出現的次數
for word in vocabulary:
    if word in count.keys():
        count[word] += 1
    if word not in count.keys():   
        count[word] = 1
count = sorted(count.items() , key = lambda x : x[1] , reverse = True) # 由詞出現的次數由高排到低
count = [('UNK' , -1)] + count
count = count[:50000] # 選出前50000個最常出現的詞
vocab_to_int = {}
for i , word in enumerate(count):
    vocab_to_int[word[0]] = i

#import collections
#count = [('UNK', -1)]
#count.extend(collections.Counter(vocabulary).most_common(vocabulary_size - 1)) # 可代替22行~34行
#vocab_to_int = {}
#for i , word in enumerate(count):
#    vocab_to_int[word[0]] = i    
    
    
# 2.把vocabulary全部換成編號  
vocabulary_transform_int = [] 
unk_count = 0
for word in vocabulary:
    if word not in vocab_to_int.keys():
        unk_count += 1
        vocabulary_transform_int.append(0) # 0代表vocab_to_int['UNK']
    elif word in vocab_to_int.keys():
        index = vocab_to_int[word]
        vocabulary_transform_int.append(index)

count = [list(content) for content in count]  # 這一步只是為了count[0][1] = unk_count能執行，因為在tuple的結構下無法改變count[0][1]       
count[0][1] = unk_count
int_to_vocab = dict(zip(vocab_to_int.values() , vocab_to_int.keys()))