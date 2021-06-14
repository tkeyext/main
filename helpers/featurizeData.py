import math
from pandas.core.frame import DataFrame
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


def getWordsFeaturised():
  """processes all the tokenized words and replaces them with feature values"""
  words = pd.read_csv("./data/words.csv")
  training_data = DataFrame(columns=["X", "Y"]).rename_axis("ID")
  
  for i in words.index:
    print("processing row", i)
    word = str(words["Word"].iloc[i])
    result = int(words["Keyword"].iloc[i]) 

    features = []
    
    #uppercasedness 
    feature_uppercased = uppercased(word) 
    features.append(feature_uppercased)

    #ngram value
    feature_ngram = ngram(word)
    features.append(feature_ngram)

    #totalfreq value
    feature_totalfreq = totalFreq(word)
    features.append(feature_totalfreq)

    #tfidf value
    document = int(words["Document"].iloc[i])
    feature_tfidf = tfidf(word, document)
    features.append(feature_tfidf)

    training_data.loc[len(training_data.index)] = [features, result]

    if i > 100:
      break
  training_data.to_csv("./data/training.csv")
  

getWordsFeaturised()