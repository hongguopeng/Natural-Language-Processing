import numpy as np
import collections

"""
skip gram與CBOW其實是差不多的                 
skip gram : 藉由"內容"去猜"上下文"
CBOW      : 藉由"上下文"去猜"內容"
"""

"""
以下以skip gram來說明:
選擇文本其中的 3 個詞，並選擇這3個詞的上文2個詞、下文2個詞

第 1 次訓練->
index : 0 , 1 , 2 
input_data:  data[0] , data[0] , data[1] , data[1] , data[1] , data[2] , data[2] , data[2] , data[2] 
output_data: data[1] , data[2] , data[0] , data[2] , data[3] , data[0] , data[1] , data[3] , data[4] 

第 2 次訓練->
index : 3 , 4 , 5
input_data:  data[3] , data[3] , data[3] , data[3] , data[4] , data[4] , data[4] , data[4] , data[5] , data[5] , data[5] , data[5] 
output_data: data[1] , data[2] , data[4] , data[5] , data[2] , data[3] , data[5] , data[6] , data[3] , data[4] , data[6] , data[7] 

第 3 次訓練->
index : 6 , 7 , 8
input_data:  data[6] , data[6] , data[6] , data[6] , data[7] , data[7] , data[7] , data[7] , data[8] , data[8] , data[8] , data[8] 
output_data: data[4] , data[5] , data[7] , data[8] , data[5] , data[6] , data[8] , data[9] , data[6] , data[7] , data[9] , data[10] 

.
.
.

第 334 次訓練->
index : 999 , 0 , 1
input_data:  data[999] , data[999] , data[0] , data[0] , data[1] , data[1] , data[1] 
output_data: data[997] , data[998] , data[1] , data[2] , data[0] , data[2] , data[3] 
"""


# data為模擬文本，總字庫有5000個，而文章從中選1000個字出來(有重複)，裡面的值都代表一個詞的編號
data = np.random.choice(5000 , size = 1000 , replace = True) 
data = list(data)
skip_window = 2 # 代表選取從這個詞的上文選2個詞，下文也選2個詞


#-------------------第1種寫法-------------------#
test = []
test_ = []
word_push = []
word_push_ = []

organize = {'input_data' : [] , 'label'  : []}
count = 0
for i in range(0 , 40): # 模擬訓練過程40次，取出每一次要訓練的batch_dat與batch_label
    
#    word_push = []
#    word_push_ = []
#    for _ in range(0 , 32):
#        word_push.append(data[count])
#        word_push_.append(count)
#        count = count + 1
#        if count == len(data):
#            count = 0
#    test.append(word_push)
#    test_.append(word_push_)
    
    input_data = []
    label = []
    for j in range(0 , 32):  # 一個batch抓32個詞，而一個詞最剁會有2個上文、2個下文
        idx = (32 * i + j) % len(data)  
        print(idx)
        label_temp = []
        for index in range(idx - skip_window ,  idx + skip_window + 1):
            if index >= 0 and index < len(data) :  # ex:在第1個詞前面不會有上文 , 在第最後一個詞後面不會有下文
                label_temp.append(data[index])
        print(label_temp)         
        label_temp.remove(data[idx])
        label.extend(label_temp) 
        
        for _ in range(0 , len(label_temp)):
            input_data.append(data[idx])  
            
    organize['input_data'].append(input_data)
    organize['label'].append(label)         
#-------------------第1種寫法-------------------#


#-------------------第2種寫法-------------------#
data = np.array(data)

index = collections.deque(np.arange(0 , 1000))

organize = {'input_data' : [] , 'label'  : []}

for i in range(0 , 40): # 模擬訓練過程40次，取出每一次要訓練的batch_dat與batch_label
    
    input_data = []    
    label = []
    for j in range(0 , 32): # 一個batch抓32個詞，而一個詞最多會有2個上文、2個下文
        label_temp = np.arange(index[j] - skip_window , index[j] + skip_window + 1)
        label_temp = label_temp[label_temp >= 0]         # ex:在第1個詞前面不會有上文
        label_temp = label_temp[label_temp < len(index)] # ex:在第最後一個詞後面不會有下文
        label_temp = list(label_temp)
        label_temp.remove(index[j])
        label.extend(list(data[np.array(label_temp)]))
        
        input_data_temp = []
        for _ in range(0 , len(label_temp)):
            input_data_temp.append(data[index[j]])
        input_data.extend(input_data_temp)   
                    
    index.rotate(-32) 
    
    organize['input_data'].append(input_data)
    organize['label'].append(label)
#-------------------第2種寫法-------------------#