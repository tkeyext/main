import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
import re

stop = stopwords.words('turkish')

def getTermFrequency (word: str, words: list) -> int:
  """Returns the term frequency of the given word in the given word list."""
  return words.count(word)

def getWordsTokenized ():
  """Processes all of the words in the raw csv file."""
  raw = pd.read_csv('./data/documents.csv')
  tokenized = DataFrame(columns=['Document', 'Word', 'Keyword']).rename_axis('ID')
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
    # Unigrams
    words = [word for word in words if word not in stop]
    words = [word for word in words if len(word) > 0]
    # Digrams
    digrams = []
    for j in range(len(words) - 1):
      digrams.append(f'{words[j]} {words[j+1]}')
    words += digrams
    # Trigrams
    trigrams = []
    for j in range(len(words) - 2):
      trigrams.append(f'{words[j]} {words[j+1]} {words[j+2]}')
    words += trigrams
    
    for word in words:
      # tokenized.loc[len(tokenized.index)] = [word, float(words.count(word) / len(words)), int(word.lower() in keywords)]
      tokenized.loc[len(tokenized.index)] = [i, word, int(word.lower() in keywords)]
    # Comment this out later
    if i > 500:
      break  
  tokenized.to_csv('./data/words.csv')


getWordsTokenized()