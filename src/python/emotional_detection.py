#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = keras.models.load_model('model/emotional_model')
with open('model/tokenizer.pickle','rb') as handle:
 tokenizer = pickle.load(handle)
with open('model/label_encoder.pickle','rb') as enc:
 index_to_class = pickle.load(enc)



def get_sequences(tokenizer,tweets):
  maxlen = 50
  sequence = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(sequence, truncating = 'post', padding = 'post', maxlen = maxlen)
  return padded


def predicting(inp):
    lists = [inp]
    padded = get_sequences(tokenizer,lists)

    result = model.predict(np.expand_dims(padded[0],axis=0))[0]

    tag = index_to_class[np.argmax(result).astype('uint8')]   
    return tag


