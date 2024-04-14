#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("netflix_titles.csv")


# In[3]:


df


# In[4]:


df["description"]


# In[5]:


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[6]:


import nltk
nltk.download('stopwords')


# In[7]:


stop_words = set(stopwords.words('english'))


# In[8]:


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return words


# In[9]:


df['processed_description'] = df['description'].apply(preprocess_text)


# In[10]:


tokens=[['love', 'like', 'politely'],
 ['life', 'romantic', 'horror', 'superhero', 'movie', 'time']]
# Import Wor2vec
from gensim.models.word2vec import Word2Vec
# By default min_count will be 5. Here we have small set of tokens so replaced count with 1.
model = Word2Vec(sentences=df['processed_description'], vector_size=100, window=5, min_count=1, workers=4)
#Save the model
model.save('word2vec')
#Load the model
model = Word2Vec.load('word2vec')


# In[11]:


vector_for_word = model.wv['movie']


# In[12]:


model.wv.most_similar("love")


# In[13]:


model.wv.most_similar("movie")


# In[14]:


model.wv.most_similar("superhero")


# In[ ]:





# In[ ]:




