import nltk
import re
import numpy as np

def document (doc):
  doc = re.sub(' \d+', " ", doc)
  doc = re.sub('[,.;:]', '', doc)
  doc = doc.lower()
  doc = doc.strip()
  tokens = nltk.WordPunctTokenizer(doc)
  tokens = [token for token in tokens if token not in nltk.corpus.stopwords.words('turkish')]
  doc = ' '.join(tokens)
  return doc

def documents (docs):
  normalizeDocuments = np.vectorize(document)
  normalizedDocuments = normalizeDocuments(docs)
  print(normalizedDocuments)