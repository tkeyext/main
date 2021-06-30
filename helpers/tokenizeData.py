from numpy import column_stack
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
  tokenized = open('./data/words', 'w')

  for i in raw.index:
    print('Processing row', i)
    # Define entities
    entities = nlp(str(raw['Body'].iloc[i])).ents
    entities = list(filter(lambda X: X.label_ != "CARDINAL" and X.label_ != "DATE", entities))
    entities = list(map(lambda x: str(x).lower().strip(),entities))
    # Handle keywords
    keywords = list(str(raw['Tags'].iloc[i]).split('??'))
    keywords = list(map(lambda x: x.lower(),keywords))
    # Handle body
    body = str(raw['Body'].iloc[i]).strip()
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
      tokenized.write(f'{i},{word},{int(word.lower() in entities)},{int(word.lower() in keywords)}\n')

  tokenized = pd.read_csv('./data/words', names=["Document", "Word", "Entities", "Keyword"]).rename_axis('ID')
  tokenized.to_csv('./data/words.csv')

# getWordsTokenized()