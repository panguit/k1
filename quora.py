import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Quora/train.csv')
data.shape

from sklearn.model_selection import train_test_split

train,test=train_test_split(data,test_size=0.1)

### TEXT ANALYSIS
text=list(test.question_text)

def maxlen (text):
    #INPIT : text - list of sentences
    #OUTPUT : max number of words
    res = 0
    for q in text:
        nw = len(q.split())
        if nw > res:
            res = nw

    return res

n=maxlen(text)

c=0
for q in text :
    c+=1
    if(len(q.split()) == n):
        print(q+'\n')
        print(c)




### WORD EMBEDDING


