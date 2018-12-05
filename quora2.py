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


### SPLIT TRAIN TEST
def split(embed,Y,test_size=0.3):
    index = np.random.rand(len(embed)) > test_size
    embed_train = embed[index]
    embed_test = embed[~index]
    Y_train = Y[index]
    Y_test = Y[~index]
    return (embed_train,Y_train,embed_test,Y_test)

embed_train,Y_train,embed_test,Y_test = split(embed,Y)


# 50:50 FOR TARGET
def mixing50 (embed_train,Y_train) :
    index= Y_train==1

    embed_1 = embed_train[index]
    embed_0 = embed_train[~index]
    embed_0 = embed_0[0:len(embed_1)]

    Y1=Y_train[index]
    Y0=Y_train[~index]
    Y0=Y0[0:len(Y1)]

    emb_samp=np.concatenate( (embed_1,embed_0) )
    Y_samp = np.concatenate( (Y1,Y0) )
    return (emb_samp,Y_samp)

x_train,y_train=mixing50(embed,Y)

#shuffling data
from sklearn.utils import shuffle
x_train = shuffle(x_train,random_state=42)
y_train = shuffle(y_train,random_state=42)
y_train=y_train.reshape( (len(y_train),1))


#### MODELLING
from keras.models import Sequential
from keras.layers import Dense,LSTM,CuDNNLSTM,BatchNormalization
from sklearn.metrics import confusion_matrix,accuracy_score

mod=Sequential()
#mod.add( BatchNormalization())
mod.add(LSTM(128, input_shape= x_train.shape[1:],return_sequences=  True,activation='relu'))
mod.add( BatchNormalization())
#mod.add(LSTM(32,activation='relu',return_sequences=True))
mod.add(LSTM(32,activation='relu'))
mod.add(Dense(1,activation='sigmoid'))

mod.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#mod.fit(embed, Y)


#BALANCE WEIGHT

#class_weights = {0: 1, 1: 15}
mod.fit(x_train, y_train ,batch_size=10, epochs= 3)
#mod.fit(embed,Y,class_weight= class_weights)

confusion_matrix(y_train,mod.predict_classes(x_train))

x_test = embed[(len(embed)-1000):len(embed)]
y_test = Y[(len(embed)-1000):len(embed)]
y_test.mean()
y_pred = mod.predict_classes(x_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)




















#TEST MODEL ON MNIST SET
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(xm,ym),(xt,yt) = mnist.load_data()

mod1=Sequential()
#mod.add( BatchNormalization())
mod1.add(LSTM(32, input_shape= xm.shape[1:],return_sequences=  True,activation='relu'))
mod1.add( BatchNormalization())
mod1.add(LSTM(32,activation='relu'))
mod1.add(Dense(10,activation='softmax'))
mod1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mod1.fit(xm, ym )