# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import string
from textblob import TextBlob
import re

trn=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

tst=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')




#Data Clean

def remove_punc(tweet):
        new_tweet=[]
        punc_removed=[]
        punc_removed_join=[]
        punc_removed = [char for char in tweet if char not in string.punctuation]
        punc_removed_join = ''.join(punc_removed).lower()
        new_tweet=''.join(punc_removed_join)
        
        return new_tweet 
        
# Let's remove punctuations from our dataset 
trn['text'] = trn['text'].apply(remove_punc)
tst['text'] = tst['text'].apply(remove_punc)

trn


# stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#def process(text):
    
#    for line in text:
#            words=nltk.word_tokenize(line)
#            new_line=[wo for wo in words if wo not in stop_words]
#            line_joined=' '.join(new_line)
#            result.append(line_joined)
#            print(new_line,line_joined,result)
            
 #   return result

def tokenize(text):
    split=re.split("\W+",text) 
    return split
trn['text']=trn['text'].apply(lambda x: tokenize(x.lower()))
trn.head()

def remove_stopwords(text):
    text=[word for word in text if word not in stop_words]
    return text
trn['text'] = trn['text'].apply(lambda x: remove_stopwords(x))
trn.head()

X=trn['text']
X_test=tst['text']


Y=trn['target']
X

X=print([' '.join(c for c in list(lst)) for lst in X])

#training
tfi=TfidfVectorizer()

tfi.fit(X)

X=tfi.transform(X)
X_test=tfi.transform(X_test)

svm=SVC()
svm.fit(X,Y)

pred=svm.predict(X_test)


pred=pd.DataFrame(pred)

# %% [code]
id_col=tst.iloc[:,0]
id_col

# %% [code]
pred.insert(0,'id',id_col)

# %% [code]
pred.columns=['id','target']

# %% [code]
pred.to_csv('/kaggle/working/submission.csv', index = True)
