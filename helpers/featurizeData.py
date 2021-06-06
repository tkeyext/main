import math
from pandas.core.indexes import base
from wordfreq import word_frequency as wf
import pandas as pd
import math

def uppercased (word: str) -> int:
  """Returns 1 if word is uppercased at some point."""
  return int(word == word.lower())

def ngram (word: str) -> int:
  """Returns the ngram value of the word"""
  return int(len(word.split()))

def totalFreq (word: str) -> float:
  """Returns the total frequency of the word among all."""
  max = wf('ve', 'tr') # Assing the most common word the value 1
  return float(wf(word, 'tr') / float(max))

def tfidf (word: str, document: int) -> float:
  """Returns the tfidf value of the given word."""
  documents = pd.read_csv('./data/documents.csv')
  words = pd.read_csv('./data/words.csv')
  docwords = pd.DataFrame(words[words['Document'] == document])
  tf = len(pd.DataFrame(docwords[docwords['Word'] == word])) / len(docwords)
  print('TF is', tf)
  docswithword = pd.DataFrame(words[words['Word'] == word])
  unique = docswithword["Document"].value_counts().to_frame('n').rename_axis('Document')
  idf = math.log(len(documents) / len(unique))
  print('IDF is', idf)
  tfidf = float(tf * idf)
  print('tfidf is', tfidf)
  return tfidf

tfidf('geçtiğimiz', 68)

