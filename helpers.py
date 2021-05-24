import re
from nltk.corpus import stopwords
from pandas.core.frame import DataFrame
from wordfreq import word_frequency as wf
import pandas as pd
import math

def removePunctuations(text: str) -> str:
  """Strips the input from non-word and non-digit characters."""
  text = re.sub('[^\w\d ]', '', text)
  return text

def removeStopwords(text: str) -> str:
  """Stripts the input from stopwords."""
  stop_words = stopwords.words('turkish')
  words = text.lower().split()
  filtered = [word for word in words if word not in stop_words]
  return ' '.join(filtered)

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

def getTotalFrequency (word: str) -> float:
  """Returns the frequency of input word over the most frequent word that exists in the language."""
  max = wf('ve', 'tr')
  return float(wf(word, 'tr') / float(max))

def getUppercased (text: str) -> int:
  """Returns 1 if text is uppercased at some point."""
  lowered = text.lower()
  return int(text != lowered)

def getBoW (text: str) -> list:
  """Returns the list of bag of words from text."""
  normalized = removeStopwords(removePunctuations(text))
  return normalized.split()

def getWordSet (documents: list) -> set:
  """Returns the complete for set for documents"""
  wordSet = set()
  for document in documents:
    bow = getBoW(document)
    wordSet = wordSet.union(set(bow))
  return wordSet

def getWordDict (documents: list) -> dict:
  """Returns the complete dictionary for all of the occured words throughput the documents."""
  wordSet = getWordSet(documents)
  wordDict = dict.fromkeys(wordSet, 0)
  return wordDict

def populateWordDict (wordDict: dict, text: str) -> dict:
    populated = wordDict.copy()
    for word in getBoW(text):
      populated[word] += 1
    return populated

def getTermFrequencies (text: str, wordDict: dict) -> dict:
  tf = {}
  bowLength = len(getBoW(text))
  words = populateWordDict(wordDict, text)
  for word, count in words.items():
    tf[word] = count / float(bowLength)
  return tf

def getInverseDocumentFrequencies (documents: list, wordDict: dict) -> dict:
  idf = {}
  documentsCount = len(documents)
  idf = dict.fromkeys(populateWordDict(wordDict, documents[0]).keys(), 0)
  for document in documents:
    docWordDict = populateWordDict(wordDict, document)
    for word, freq in docWordDict.items():
      if freq > 0:
        idf[word] += 1
  
  for word, freq in idf.items():
    idf[word] = math.log10(documentsCount / float(freq))

  return idf

def getTFIDFs (documents: list) -> DataFrame:
  wordDict = getWordDict(documents)
  idfs = getInverseDocumentFrequencies(documents, wordDict)
  tfidfs = list()
  for document in documents:
    tfidf = {}
    tf = getTermFrequencies(document, wordDict)
    # print('TF of', document, 'is:\n', tf)
    for word, freq in tf.items():
      tfidf[word] = freq * idfs[word]
    tfidfs.append(tfidf)
  return pd.DataFrame(tfidfs)

def main ():
  S1 = "Bir araba geldi."
  S2 = "Kırmızı bir araba geldi."
  S3 = "Kırmızı araba geldi."
  S4 = "Kırmızı bisiklet geldi."

  df = getTFIDFs ((S1, S2, S3, S4))
  print(df.head())
  # print(df.head())

main()