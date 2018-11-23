import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data=pd.read_csv('titanic/train.csv')
Y_train=np.array(data['Survived'])
X_train=np.array(data)

#Delete name
X_train=np.delete(X_train,(3,8,10),1)
test=[X_train[:,3]=='male']
X_train[:,:-1]=test

model=Sequential()
model.add((Dense(10,activation='relu')))
model.add((Dense(5,activation='relu')))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=4)

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()
X_train_log=X_train[np.logical_not(np.isnan(X_train))]
X_train_log=X_train_log.reshape((-1,3))

logreg.fit(X_train_log,Y_train)
