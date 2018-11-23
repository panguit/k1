#Kaggle competition
# THE ACTUAL VIRTUAL.ENV WILL BE USED FOR OTHER DS/AI PROJECT

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import matplotlib.pyplot as plt

data=pd.read_json("all/train.json")
data.columns

audios=[k for k in data['audio_embedding']]
Y=np.asanyarray(data['is_turkey'])


def split(data):
    train,test=train_test_split(data,test_size=0.3)

    Train_audios = [k for k in train['audio_embedding']]
    Train_Y = np.asanyarray(train['is_turkey'])

    Test_audios = [k for k in test['audio_embedding']]
    Test_Y = np.asanyarray(test['is_turkey'])

    # CORRECT SHAPE OF AUDIOS
    Train_audios = pad_sequences(Train_audios, maxlen=10)
    Test_audios = pad_sequences(Test_audios, maxlen=10)

    # NORMALIZATION
    Train_audios = tf.keras.utils.normalize(Train_audios)
    Test_audios = tf.keras.utils.normalize(Test_audios)

    return (Train_audios,Train_Y,Test_audios,Test_Y)

Train_audios,Train_Y,Test_audios,Test_Y = split(data)




from keras.models import Sequential
from keras.layers import Dense,Flatten


model=Sequential()
model.add(Flatten())
model.add(Dense(128,activation='relu'))
#model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
#model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
#model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#COMPILE
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#LEARNING
model.fit(Train_audios,Train_Y,batch_size=4,epochs=3)

Test_predict=model.predict(Test_audios)
Test_Y_pred=np.asarray([Test_predict>0.5])
Test_Y_pred=Test_Y_pred.reshape(Test_Y.shape)

from sklearn.metrics import confusion_matrix

confusion_matrix(Test_Y,Test_Y_pred)



##### SUBMIT DATA
test_data=pd.read_json('all/test.json')
test_audio=test_data['audio_embedding']
test_audio=pad_sequences(test_audio,maxlen=10)
test_audio=tf.keras.utils.normalize(test_audio)

testpred=model.predict(test_audio)
testpred=testpred.reshape(1196)
testypred=np.asarray([testpred>0.5])
testypred=testypred.reshape(1196)

test_id=test_data['vid_id']
test_res=np.stack( (test_id,testpred))
test_res=pd.DataFrame(test_res)
test_res=pd.DataFrame.transpose(test_res)
test_res.columns=['vid_id','pred']

sub=pd.read_csv('all/sample_submission.csv')

r=pd.merge(test_res,sub,on='vid_id')

r['is_turkey']=np.where(r['pred']>0.5,1,0)
r1=r.drop('pred',axis=1)

r1.to_csv('res',index=False)



















######################################################################3
def extractinfo (data):
    audios=[]
    res=[]
    for i in range(len(data)):
        # info DATA
        tmpsample=data[i]
        Yturkey=tmpsample["is_turkey"]
        id=tmpsample["vid_id"]
        res.append([id,Yturkey])

        # audio DATA
        audio=tmpsample["audio_embedding"]
        audios.append(audio)
    return (np.asarray(res),audios)

datainfo,audios = extractinfo(data)

audio.read()

res=[]
for i in range(len(audios)):
    audio_samp=[]
    audio=audios[i]
    for j in range(len(audio)):
        audio_samp.append(audio[j])
    res.append(audio_samp)





def extractaudio (audios) :
    res=[]
    for i in range(len(audios)):
        sample=audios[i]
        sample=np.asarray(sample)
        sample=sample.reshape(sample.shape[0]*sample.shape[1])
        res.append(sample)
    return res

audios=extractaudio(audios)
test=np.array(audios)

Y_train=datainfo[:,1]
Y_train=Y_train.astype(np.int)
Y_train=Y_train.reshape(1195)

X_train=audios






##LETS DO SOME NEURALNET


from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=1)