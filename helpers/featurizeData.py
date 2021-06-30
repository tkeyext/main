import math
from pandas.core.frame import DataFrame
from pandas.core.indexes import base
from wordfreq import word_frequency as wf
import pandas as pd
import math
import spacy
# import en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def uppercased (word: str) -> int:
  """Returns 1 if word is uppercased at some point."""
  return int(word != word.lower())

def ngram (word: str) -> int:
  """Returns the ngram value of the word"""
  return int(len(word.split()))

# def totalFreq (word: str) -> float:
#   """Returns the total frequency of the word among all."""
#   max = wf('ve', 'tr') # Assing the most common word the value 1
#   return float(wf(word, 'tr') / float(max))

def tfidf (docsLen: int, words: DataFrame, docwords: DataFrame, word: str) -> float:
  """Returns the tfidf value of the given word."""
  tf = len(pd.DataFrame(docwords[docwords['Word'] == word])) / len(docwords)
  docswithword = pd.DataFrame(words[words['Word'] == word])
  unique = docswithword["Document"].value_counts().to_frame('n').rename_axis('Document')
  if len(unique) == 0:
    idf = math.log((docsLen + 1) / 1)
  else:
    idf = math.log(docsLen / len(unique))
  tfidf = float(tf * idf)
  return tfidf

def namedEntity (word: str) -> int:
  """Returns 1 if word is a named entity"""
  return nlp(word)


def getWordsFeaturised():
  """processes all the sampled tokenized words and replaces them with feature values"""
  words = pd.read_csv("./data/sampled.csv").sample(frac=1)

  tfidf_documents = pd.read_csv('./data/documents.csv')
  tfidf_documents_length = len(tfidf_documents)
  tfidf_words = pd.read_csv('./data/words.csv')

  training = open('./data/training', 'w')
  
  for i in words.index:
    print('Processing row', i)
    word = str(words["Word"].iloc[i])
    result = int(words["Keyword"].iloc[i]) 

    features = []
    
    #uppercasedness 
    feature_uppercased = uppercased(word) 
    features.append(feature_uppercased)

    #ngram value
    feature_ngram = ngram(word)
    features.append(feature_ngram)

    #tfidf value
    document = int(words["Document"].iloc[i])
    tfidf_docwords = pd.DataFrame(tfidf_words[tfidf_words['Document'] == document])
    feature_tfidf = tfidf(tfidf_documents_length, tfidf_words, tfidf_docwords, word)
    features.append(feature_tfidf)

    #entitiy value 
    entitization = int(words["Entities"].iloc[i])
    features.append(entitization)

    # features = [features]
    training.write(f'"{features}",{result}\n')

  training = pd.read_csv('./data/training', names=["X", "Y"]).rename_axis('ID')
  training.to_csv('./data/training.csv')
    
# getWordsFeaturised()
