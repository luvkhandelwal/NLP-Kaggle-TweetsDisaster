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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/nlp-getting-started/train.csv
/kaggle/input/nlp-getting-started/test.csv
/kaggle/input/nlp-getting-started/sample_submission.csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

trn=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

tst=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

X=trn['text']
Y=trn['target']
X_test=tst['text']

tfi=TfidfVectorizer(max_features=12000)

X=tfi.fit_transform(X)
X_test=tfi.transform(X_test)

clf=DecisionTreeClassifier()
clf.fit(X,Y)

pred=clf.predict(X_test)
pred=pd.DataFrame(pred)

id_col=tst.iloc[:,0]
id_col

pred.insert(0,'id',id_col)

pred.columns=['id','target']

pred.to_csv('/kaggle/working/submission.csv', index = True) 
