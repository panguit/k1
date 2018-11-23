import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

plt.imshow(X_train[0],cmap=plt.cm.binary)

X_train=tf.keras.utils.normalize(X_train)
X_test=tf.keras.utils.normalize(X_test)

plt.imshow(X_train[0],cmap=plt.cm.binary)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=1)

predict=model.predict(X_test)
class1=np.argmax(predict[0])
