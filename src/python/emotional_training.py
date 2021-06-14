#!/usr/bin/env python
# coding: utf-8

# ## Tweet Emotion Recognition: Natural Language Processing with TensorFlow
# 
# ---
# 
# Dataset: [Tweet Emotion Dataset](https://github.com/dair-ai/emotion_dataset)
# 
# This is a starter notebook for the guided project [Tweet Emotion Recognition with TensorFlow](https://www.coursera.org/projects/tweet-emotion-tensorflow)
# 
# A complete version of this notebook is available in the course resources
# 
# ---
# 
# ## Task 1: Introduction

# ## Task 2: Setup and Imports
# 
# 1. Installing Hugging Face's nlp package
# 2. Importing libraries

# In[ ]:





# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random


def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

    
print('Using TensorFlow version', tf.__version__)


# ## Task 3: Importing Data
# 
# 1. Importing the Tweet Emotion dataset
# 2. Creating train, validation and test sets
# 3. Extracting tweets and labels from the examples

# In[2]:


dataset =  nlp.load_dataset('emotion')


# In[3]:


train = dataset['train']
val = dataset['validation']
test = dataset['test']


# In[4]:


def get_tweet(data):
  tweets = [x['text']for x in data]
  labels = [x['label']for x in data]
  return tweets, labels


# In[5]:


tweets,labels = get_tweet(train)


# In[6]:


tweets[0],labels[0]


# In[ ]:





# ## Task 4: Tokenizer
# 
# 1. Tokenizing the tweets

# In[7]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[8]:


tokenizer = Tokenizer(num_words=10000, oov_token ='<UNK>')
tokenizer.fit_on_texts(tweets)


# In[9]:


tokenizer.texts_to_sequences([tweets[0]])


# In[10]:


tweets[0]


# ## Task 5: Padding and Truncating Sequences
# 
# 1. Checking length of the tweets
# 2. Creating padded sequences

# In[ ]:


plt.hist(labels,bins =11)
plt.show()


# In[ ]:


lengths = [len(t.split(' ')) for t in tweets]
plt.hist(lengths, bins = len(set(lengths)))
plt.show()


# In[ ]:


maxlen = 50


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


print(type(tweets))


# In[ ]:


def get_sequences(tokenizer,tweets,maxlen=50):
  sequence = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(sequence, truncating = 'post', padding = 'post', maxlen = maxlen)
  return padded


# In[ ]:


padded_train_seq = get_sequences(tokenizer,tweets)


# In[ ]:





# ## Task 6: Preparing the Labels
# 
# 1. Creating classes to index and index to classes dictionaries
# 2. Converting text labels to numeric labels

# In[ ]:


classes = set(labels)


# In[ ]:


class_to_index = dict((c,i) for i,c in enumerate(classes))
index_to_class = dict((v,k) for k,v in class_to_index.items())


# In[ ]:


class_to_index


# In[ ]:


names_to_ids = lambda labels:np.array([class_to_index.get(x) for x in labels])


# In[ ]:


train_labels = names_to_ids(labels)


# In[ ]:





# In[ ]:





# ## Task 7: Creating the Model
# 
# 1. Creating the model
# 2. Compiling the model

# In[ ]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000,16,input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6,activation='softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)


# In[ ]:


model.summary()


# ## Task 8: Training the Model
# 
# 1. Preparing a validation set
# 2. Training the model

# In[ ]:


val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer,val_tweets)
val_labels = names_to_ids(val_labels)


# In[ ]:





# In[ ]:


h = model.fit(
    padded_train_seq, train_labels,
    validation_data = (val_seq,val_labels),
    epochs = 20,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)
    ]
)


# In[ ]:





# ## Task 9: Evaluating the Model
# 
# 1. Visualizing training history
# 2. Prepraring a test set
# 3. A look at individual predictions on the test set
# 4. A look at all predictions on the test set

# In[ ]:


show_history(h)


# In[ ]:


test_tweets,test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer,test_tweets)
test_labels = names_to_ids(test_labels)


# In[ ]:


model.save("emotional_model")

import pickle
with open('tokenizer.pickle','wb') as handle:
  pickle.dump(tokenizer,handle,protocol = pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle','wb') as ecn_file:
  pickle.dump(index_to_class,ecn_file,protocol = pickle.HIGHEST_PROTOCOL)


# In[ ]:


i = random.randint(0, len(test_labels) - 1)

print('Sequence:',test_tweets[i])
print('Emotion:',index_to_class[test_labels[i]])

p = model.predict(np.expand_dims(test_seq[i],axis=0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]

print('Predicted Emotion:', pred_class)


# In[ ]:





# In[ ]:





# In[ ]:




