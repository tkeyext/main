from operator import itemgetter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
nltk.download('punkt')

word_set = set(stopwords.words('turkish'))

def getWords (text):
  words = text.split(' ')
  return words

def getSentences (text):
  sentences = nltk.tokenize.sent_tokenize(text)
  return sentences

def calculateTotalFrequency (words):
  scores = {}
  for word in words:
    word = word.replace('.', '')
    if word not in word_set:
      if word in scores:
        scores[word] += 1
      else:
        scores[word] = 1
  return scores

def checkWordAgainstSentences (word, sentences):
  final = [all([w in x for w in word]) for x in sentences]
  sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
  return int(len(sent_len))

def calculateInverseDocumentFrequency (words, sentences, sentencesLength):
  scores = {}
  for word in words:
    word = word.replace('.', '')
    if word not in word_set:
      if word in scores:
        scores[word] = checkWordAgainstSentences(word, sentences)
      else:
        scores[word] = 1
  scores.update((x, math.log(int(sentencesLength)/y)) for x, y in scores.items())
  return scores

def calculateScores (totalFrequencyScores, inverseDocumentFrequencyScores):
  scores = { 
    word: totalFrequencyScores[word] * inverseDocumentFrequencyScores.get(word, 0) for word in totalFrequencyScores.keys() 
  }
  return scores


def run (text):
  words = getWords(text)
  sentences = getSentences(text)
  totalFrequencyScores = calculateTotalFrequency(words)
  inverseDocumentFrequencyScores = calculateInverseDocumentFrequency(words, sentences, len(sentences))
  # print(totalFrequencyScores + '\n')
  scores = calculateScores(totalFrequencyScores, inverseDocumentFrequencyScores)

  result = dict(sorted(scores.items(), key = itemgetter(1), reverse = True)[:5])
  return result
  

