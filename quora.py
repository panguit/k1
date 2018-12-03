#MAX SIZE IS 135

### WORD EMBEDDING
import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Quora/train.csv')
data=data.tail(100)

#DELETE STOPWORDS
from nltk.corpus import stopwords
text=np.asarray(data.question_text)
test=[word for word in text if word not in stopwords.words('english')]

words=[]
for i in range(100):
    word_list=text[i].split()
    filt_word = [word for word in word_list if word not in stopwords.words('english')]
    words.append(filt_word)








model = gensim.models.KeyedVectors.load_word2vec_format('embedding/GoogleNews-vectors-negative300.bin', binary=True)





emb=np.zeros( (100,10,300) )

i,j = 0,0
for t in data.question_text:
    words=t.split()
    words=words[0:9]
    for w in words:
        emb[i,j,:]=model.wv[w]
        j+=1
    i+=1




