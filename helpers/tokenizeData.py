import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
import re
import spacy
nlp = spacy.load("en_core_web_sm")

stop = stopwords.words('turkish')

def getTermFrequency (word: str, words: list) -> int:
  """Returns the term frequency of the given word in the given word list."""
  return words.count(word)

def getWordsTokenized ():
  """Processes all of the words in the raw csv file."""
  raw = pd.read_csv('./data/documents.csv')
  tokenized = DataFrame(columns=['Document', 'Word', 'Entities', 'Keyword']).rename_axis('ID')
  # tokenized = pd.read_csv('./data/words.csv', names=['Document', 'Word', 'Entities', 'Keyword']).rename_axis('ID')
  for i in raw.index:
    print('Processing row', i)
    # print('\n\n', str(row['Body']))
    entities = nlp(str(raw['Body'].iloc[i])).ents
    print([(X.text, X.label_) for X in entities])
    body = str(raw['Body'].iloc[i]).strip()
    keywords = list(str(raw['Tags'].iloc[i]).split('??'))
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
    # Trigrams
    trigrams = []
    for j in range(len(words) - 2):
      trigrams.append(f'{words[j]} {words[j+1]} {words[j+2]}')
    # Append the n-grams
    words += digrams
    words += trigrams
    for word in words:
      # tokenized.loc[len(tokenized.index)] = [word, float(words.count(word) / len(words)), int(word.lower() in keywords)]
      tokenized.loc[len(tokenized.index)] = [i, word, entities, int(word.lower() in keywords)]
    # Comment this out later
    tokenized.to_csv('./data/words.csv')


getWordsTokenized()