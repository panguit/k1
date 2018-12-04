import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#NLP PACKAGE
import gensim
from nltk.corpus import stopwords
import regex as re

data=pd.read_csv('Quora/train.csv')
text=data.question_text
text=np.asarray(text)
model = gensim.models.KeyedVectors.load_word2vec_format('embedding/GoogleNews-vectors-negative300.bin', binary=True)

word_list=[]
sent_len=[]
for w in text:
    word = re.sub('\W+',' ', w)
    word = word.split()
    word = [w for w in word if w in model.wv]
    sent_len.append(len(word ))
    word_list.append(word)

#SAVE AS FILE
import pickle
with open('word_list.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(word_list, f)

with open('word_list.pkl') as f:
    word_list =pickle.load(f)

# NB WORD x TARGET ANALYSIS
sent_len = pd.DataFrame(sent_len)
data['sent_len']=sent_len
target_sent_len=data.groupby(by=['sent_len'])['target'].mean()
plt.plot(target_sent_len)

# COUNTING NUMBER OF WORDS
data.question_text.str.split(expand=True).stack().value_counts()