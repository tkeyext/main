import nltk
import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
import re

stop = stopwords.words('turkish')

def getWordsTokenized ():
  """Processes all of the words in the raw csv file"""
  raw = pd.read_csv('./data/raw.csv')
  tokenized = DataFrame(columns=['Word', 'Keyword'])
  for i, row in raw.iterrows():
    print('Processing row', i)
    # print('\n\n', str(row['Body']))
    body = str(row['Body']).strip()
    keywords = list(str(row['Tags']).split('??'))
    keywords = list(map(lambda x: x.lower(),keywords))
    # print('\n', keywords, '\n')
    body = re.sub("\n", " ", body)
    body = re.sub("\[http.*\]", "", body)
    body = re.sub("[^\w ]+", " ", body)
    words = body.split(' ')
    words = [word for word in words if word not in stop]
    words = [word for word in words if len(word) > 0]
    for word in words:
      tokenized.loc[len(tokenized.index)] = [word, int(word.lower() in keywords)]
    # Comment this out later
    if i > 100:
      break  
  tokenized.to_csv('./data/tokenized.csv')




getWordsTokenized()