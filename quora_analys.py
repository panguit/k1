import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#NLP PACKAGE
import gensim
from nltk.corpus import stopwords
import regex as re
import pickle

data=pd.read_csv('Quora/train.csv')
text=data.question_text
text=np.asarray(text)
#model = gensim.models.KeyedVectors.load_word2vec_format('embedding/GoogleNews-vectors-negative300.bin', binary=True)

def wordlist (text,model):
    #INPUT : TEXT AND THE EMBEDDING MODEL
    #OUTPUT : LIST OF LIST SENTENCExWORD AND LENGTH OF SENTENCES
    word_list=[]
    sent_len=[]
    for w in text:
        word = re.sub('\W+',' ', w)
        word = word.split()
        word = [w for w in word if w in model.wv]
        sent_len.append(len(word ))
        word_list.append(word)
    return (word_list,sent_len)

word_list,  = wordlist(text,model)

#SAVE WORD_LIST AS FILE

#with open('word_list.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(word_list, f)

#with open('word_list.pkl','rb') as f:
#    word_list =pickle.load(f)


# COUNTING NUMBER OF WORDS
#common_words = data.question_text.str.split(expand=True).stack().value_counts()
#plt.plot(np.asarray(common_words))

# JOIN WORDS AS SENTENCE
def wordlist2sentences (word_list):
    sentences=[]
    for i in range(len(word_list)):
        sentences.append(' '.join(word_list[i]))
    return sentences

sentences = wordlist2sentences((word_list))
common_words = pd.Series(sentences).str.split(expand=True).stack().value_counts()

plt.plot(common_words.values[0:500])

# DELETE 100 MOST COMMON WORDS

filterword=common_words.index[0:99]

def filter_wordlist(word_list,filterword):
    for i in range(len(word_list)):
        s=word_list[i]
        word_list[i]=[word for word in s if word not in filterword]
    return word_list

word_list_filtered=filter_wordlist(word_list,filterword)

cleaned_data = {
    'qid' : data.qid,
    'word_list' : word_list_filtered,
    'filterword' :  filterword,
    'target' : data.target
}

with open('cleaned_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(cleaned_data, f)

