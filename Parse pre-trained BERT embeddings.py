import string # Used to remove stopwords
import nltk
from nltk.corpus import stopwords
import gensim
import json
import pickle
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec
import string # Used to remove stopwords
from nltk.corpus import stopwords
import json

import torch
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords
from collections import namedtuple
from collections.abc import Mapping

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
import cuda

# Download stopwords
nltk.download('stopwords')

# Load the data
questions = pd.read_csv("questions.csv")

# Define text processing functions
def column_to_lower(df, column):
    return df[column].str.lower()

def column_remove_punctuation(df, column):
    return df[column].str.replace('[{}]'.format(string.punctuation), '', regex=True)

def column_remove_stop_words(df, column, stopwords):
    print(f"Currently processing the column: {column}")
    return df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

def remove_references(text):
    return re.sub(r"\[\d+\]", "", text)

def remove_newlines(text):
    return text.replace("\n", "")

def remove_links(text):
    return re.sub(r"https?://\S+", "", text)

# Parse stop words
stop_words = stopwords.words('english')
custom_stop_words = ["eu", "european", "commission", "member", "can", "view", "submitted", "rights", "states", "state", "ensure", "union"]
stop_words.extend(custom_stop_words)

# Clean the text
questions['text'] = column_to_lower(questions, 'text')
questions['text'] = column_remove_punctuation(questions, 'text')
questions['text'] = column_remove_stop_words(questions, 'text', stop_words)
questions['text'] = questions['text'].apply(remove_references)
questions['text'] = questions['text'].apply(remove_newlines)
questions['text'] = questions['text'].apply(remove_links)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Check if CUDA is available and move model to GPU if possible
device = torch.device('cuda')
model.to(device)

# Function to get BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling over the sequence length dimension to get a single 768-dimensional vector
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    #print(f"Embedding shape for text '{text}': {embeddings.shape}")  # This will confirm the shape is (768,)
    return embeddings


n = questions.shape[0]
#n=5
embedddings_list = []
for i in range(n):
  text_data = questions['text'][i]
  embeds = get_bert_embeddings(text_data)
  embedddings_list.append(embeds)
  print(f"Embedding {i} ready")

df = pd.DataFrame(embedddings_list)
df.to_csv("/content/drive/My Drive/Embeddings_BERT_final_test.csv")
