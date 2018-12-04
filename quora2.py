#AFTER QUORA_ANALYS
import pickle
import pandas as pd
import gensim
import numpy as np

#IMPORT WORD2VEC MODEL
model =  gensim.models.KeyedVectors.load_word2vec_format('embedding/GoogleNews-vectors-negative300.bin', binary=True)

with open('cleaned_data.pkl','rb') as f:
    cleaned_data =pickle.load(f)
f.close()
del f

data=pd.DataFrame()
data['qid']=cleaned_data['qid']
data['target']=cleaned_data['target']
word_list = cleaned_data['word_list']

del cleaned_data


#DEL INDIV BECAUSE OF MEMORY
size=100000
data=data.head(size)
word_list=word_list[0:size]


def embedding(word_list,model):
    #embedding with W2V model

    max_sentence=max([len(s) for s in word_list])
    n = len(word_list)
    emb_dim=300 #size of embedding feature

    embed=np.zeros( (n,max_sentence,emb_dim) )

    for i in range(len(word_list)):
        s=word_list[i]
        for j in range(len(s)):
            word= s[j]
            embed[i,j,:] = model.wv[word]

    return embed

embed= embedding(word_list,model)

Y=np.asarray(data.target)

del model





#### MODELLING
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import normalize

X=normalize(embed)
mod=Sequential()
mod.add(LSTM(1, input_shape= embed.shape[1:]))

mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#mod.fit(embed, Y)


#BALANCE WEIGHT

class_weights = {0: 1,
                1: 15}
mod.fit(embed, Y, class_weight=class_weights)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y,mod.predict_classes(embed))
