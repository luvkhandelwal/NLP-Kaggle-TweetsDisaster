import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

layers=keras.layers
from sklearn.preprocessing import LabelEncoder

tst=pd.read_csv('test.csv')

trn=pd.read_csv('train.csv')



trn.shape

#Training features
train_text=trn['text']

#Train labels
train_label=trn['target']

#Test features
test_text=tst['text']

#Tokenize
vocab_size=10000
token=keras.preprocessing.text.Tokenizer(num_words=vocab_size,oov_token="new")
token.fit_on_texts(train_text)


#Bag of words 
bow_train=token.texts_to_sequences(train_text)
bow_test=token.texts_to_sequences(test_text)

#Padding
pad_train=keras.preprocessing.sequence.pad_sequences(bow_train,padding='post',maxlen=30)
pad_test=keras.preprocessing.sequence.pad_sequences(bow_test,padding='post',maxlen=30)

pad_train=pd.DataFrame(pad_train)
pad_test=pd.DataFrame(pad_test)
pad_train

#Model
model=keras.Sequential()
model.add(layers.Embedding(vocab_size,16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(64,activation=tf.nn.relu))
model.add(layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


model.fit(x=pad_train,y=train_label,epochs=40,batch_size=500)

pred=model.predict(pad_test)

pred=np.where(pred>0.68,1,0)

pred=pd.DataFrame(pred)
pred





