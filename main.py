#Kaggle competition
# THE ACTUAL VIRTUAL.ENV WILL BE USED FOR OTHER DS/AI PROJECT

import json
import numpy as np
import pandas as pd


data=json.load(open("all/train.json"))

sample=data[0]
sample_audio=sample['audio_embedding']

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