import pandas as pd
from pandas import DataFrame
import math

def getBoW (text: str) -> list:
  """Returns the list of bag of words from text."""
  return text.split()

def getWordSet (documents: list) -> set:
  """Returns the complete for set for documents"""
  wordSet = set()
  for document in documents:
    bow = getBoW(document)
    wordSet = wordSet.union(set(bow))
  # print(wordSet)
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
  """This method populates the BoW and calculate the tf/idf values of each word"""
  wordDict = getWordDict(documents)
  print('Word dict:\n', wordDict)
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