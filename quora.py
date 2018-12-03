#MAX SIZE IS 135

### WORD EMBEDDING
import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Quora/train.csv')
size=100000
data=data.tail(size)

# CLEANING TEXT DATA
from nltk.corpus import stopwords
import regex as re

def wordlist (data,size):
    #INPUT : data : pandas dataframe witch question_text
    #        size : size of document
    #OUTPUT : list of list of words without stopword, special character
    text = np.asarray(data.question_text)
    words=[]
    for i in range(size):
        word_list=re.sub('\W+',' ', text[i]).split()
        filt_word = [word for word in word_list if word not in stopwords.words('english')]
        words.append(filt_word)
    return words

word_mat=wordlist(data,size)



### IMPORTING EMBEDDING DATA
# WORD2VEC  MODEL ABOUT 5G RAM
model = gensim.models.KeyedVectors.load_word2vec_format('embedding/GoogleNews-vectors-negative300.bin', binary=True)


def embevt (word_mat,word_num,model) :
    #INPUT : word_mat : matrix in line sentence (the question) and column the  words of the sentence
    #        word_num : number of word considered in LSTM model
    #        model : embedding model (with word2vec we have 300 features)
    #OUTPUT : 3 dimensional array for each question the embedding of each word
    size=len(word_mat)
    res = np.zeros( (size,word_num,300))
    for i in range(size):
        count_w=0
        for word in word_mat[i][:]:
            if count_w >= word_num:
                break
            #print(i,count_w)
            if word in model.wv :
                res[i,count_w,:]=model.wv[word]
            count_w+=1
    return res

word_emb=embevt(word_mat,3,model)

Y = np.asarray(data.target)

#FREE MEMORY
#del model

### LETS DO LSTM MODEL
from keras.models import Sequential
from keras.layers import Dense,LSTM

mod=Sequential()
mod.add(LSTM(32,activation='relu',return_sequences=True,input_shape=(3,300)))
mod.add(LSTM(32,activation='relu',return_sequences=True))
mod.add(LSTM(32,activation='relu',return_sequences=True))
mod.add(LSTM(1,return_sequences=False))

#METRICS

#CUSTUM METRICS
import tensorflow as tf
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

recall = as_keras_metric(tf.metrics.recall)
precision = as_keras_metric(tf.metrics.precision)
mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',precision])
mod.fit(word_emb, Y)



def f1_loss(y_true, y_pred):
    """Custom f1 loss for bicategorical
    y must be of shape where y.shape[-1] == 2
    y[..., 0] must be the category for true
    y[..., 1] must be the category for false
    """
    true_truth = K.dot(y_true, K.constant([1., 0.], dtype='float32', shape=(2, 1)))
    true_false = K.dot(y_true, K.constant([0., 1.], dtype='float32', shape=(2, 1)))

    y_false = K.constant(1., dtype='float32') - y_true

    fake_truth = K.dot(y_false, K.constant([1., 0.], dtype='float32', shape=(2, 1)))
    fake_false = K.dot(y_false, K.constant([0., 1.], dtype='float32', shape=(2, 1)))

    TP_temp = K.sum(true_truth * y_pred)
    FP_temp = K.sum(fake_truth * y_pred)
    FN_temp = K.sum(fake_false * y_pred)

    loss = (FP_temp + FN_temp) / (2 * TP_temp + FP_temp + FN_temp + K.epsilon())

    return loss

mod.compile(optimizer='adam', loss=f1_loss, metrics=['accuracy',precision])
mod.fit(word_emb, Y)



#CONFUSION matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(Y,mod.predict_classes(word_emb))














################TEST VALUE########################
submission=pd.read_csv('Quora/sample_submission.csv')
testcsv=pd.read_csv('Quora/test.csv')

test_wordmat=wordlist(testcsv,len(testcsv))
