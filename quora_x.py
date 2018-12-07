### TRY TO DEVELOP END TO END

import pandas as pd
import numpy as np
#NLP PACKAGE
import regex as re
import pickle

### IMPORT DATA
data=pd.read_csv('Quora/train.csv')
samp=data.sample(n=10000,random_state=42)
#del data

### IMPORT EMBEDDING MODEL
model = gensim.models.KeyedVectors.load_word2vec_format('NLP embedding/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)

def wordlist (text,model):
    #INPUT : TEXT AND THE EMBEDDING MODEL
    #OUTPUT : LIST OF LIST SENTENCExWORD AND LENGTH OF SENTENCES
    word_list=[]
    #sent_len=[]
    for w in text:
        word = re.sub('\W+',' ', w)
        word = word.split()
        word = [w for w in word if w in model.wv]
        #sent_len.append(len(word ))
        word_list.append(word)
    return word_list


word_list = wordlist(samp.question_text,model)

def wordlist2sentences (word_list):
    sentences=[]
    for i in range(len(word_list)):
        sentences.append(' '.join(word_list[i]))
    return sentences

sentences = wordlist2sentences(word_list)
common_words = pd.Series(sentences).str.split(expand=True).stack().value_counts()

def filter_wordlist(word_list,filterword):
    for i in range(len(word_list)):
        s=word_list[i]
        word_list[i]=[word for word in s if word not in filterword]
    return word_list

filterword=common_words.index[0:99]

word_list_filtered=filter_wordlist(word_list,filterword)

######## PRE PROCESSING DATA


def embedding(word_list,model,feature_size=0):
    #embedding with W2V model

    max_sentence = feature_size
    if feature_size == 0 :
        max_sentence = max([len(s) for s in word_list])

    n = len(word_list)

    emb_dim=300 #size of embedding feature

    embed=np.zeros( (n,max_sentence,emb_dim) )

    for i in range(len(word_list)):
        s=word_list[i]
        for j in range(len(s)):
            if j < max_sentence:
                word= s[j]
                embed[i,j,:] = model.wv[word]

    return embed

embed= embedding(word_list_filtered,model)

#del model

### SELECT TRAIN EXEMPLE
indiv = np.random.random_sample(10000) >0.7

x_test = embed[indiv]
y_test = samp.target[indiv]

x_train = embed[~indiv]
y_train = samp.target[~indiv]





### MODELLING

from keras.models import Sequential
from keras.layers import Dense,LSTM,CuDNNLSTM,BatchNormalization
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score

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

class_wei = {0: 1, 1: 10}
mod.fit(x_train, y_train ,batch_size=10, class_weight=class_wei,epochs=5)
#mod.fit(embed,Y,class_weight= class_weights)

confusion_matrix(y_train,mod.predict_classes(x_train))


### TESTING DATA
pred=mod.predict_classes(x_test)

confusion_matrix(y_test,pred)
accuracy_score(y_test,pred)
precision_score(y_test,pred)


######## TRY OUT WITH SUBMISSION DATA
subdata=pd.read_csv('Quora/test.csv')

sub_wordlist= wordlist(subdata.question_text,model)

#sub_wordlist_filtered= filter_wordlist()...

sub_embed= embedding(sub_wordlist,model,feature_size=34)

sub_pred=mod.predict_classes(sub_embed)

sample_submission = pd.read_csv('Quora/sample_submission.csv')
sample_submission.prediction = sub_pred

sample_submission.to_csv('Quora/submission_res.csv',index=False)