from pandas.core.frame import DataFrame
from model.regression import LogisticRegression
from nltk.corpus import stopwords
import spacy
import re
import helpers.featurizeData as featurize
import pandas as pd


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
  

# training_data = pd.read_csv("./data/training.csv")

# X = pd.DataFrame(training_data['X'].apply(lambda x: eval(x), 0).tolist(), columns=['a', 'b', 'c', 'd']).to_numpy()
# y = training_data['Y'].astype('int').to_numpy()

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

# clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# print(clf.predict(x_test[:2, :]))

# print(clf.predict_proba(x_test[:2, :]))

# print(clf.score(x_test, y_test))



def extractKeywords (body: str, regressor: LogisticRegression) -> list:
  # Extract entities
  nlp = spacy.load("en_core_web_sm")
  entities = list(
    map(
      lambda x: str(x).lower().strip(),
      list(filter(lambda x: x.label_ != 'CARDINAL' and x.label_ != 'DATE', nlp(body).ents))
    )
  )
  # Extract words from normalized body
  words = re.sub("[^\w ]+", " ",
    re.sub("\[http.*\]", "", 
      re.sub("\n", " ",
        body.strip()
      )
    )
  ).split()
  # Generate n-grams
  stop = stopwords.words('turkish')
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
  # Combine the ngrams
  words += digrams
  words += trigrams

  # Initialize tfidf background
  tfidf_documents = pd.read_csv('./data/documents.csv')
  tfidf_documents_length = len(tfidf_documents)
  tfidf_words = pd.read_csv('./data/words.csv')
  # tfidf_docwords = words
  tfidf_docwords = pd.DataFrame(words, columns=['Word']).rename_axis('ID')
  # Initialize the result
  wordsWithFeatures = []
  for word in words:
    # Organize the features
    features = [
      featurize.uppercased(word),
      featurize.ngram(word),
      featurize.tfidf(tfidf_documents_length, tfidf_words, tfidf_docwords, word),
      int(word.lower() in entities)
    ]
    print(word, features)
    # print(word, features)
    wordsWithFeatures.append(features)

  results = regressor.predict(wordsWithFeatures)
  # print(results)
  keywords = []
  print(results)
  for i in range(len(results)):
    result = results[i]
    if (result == 1):
      # print(f'Adding word ({words[i]}) because it\'s result is {result}')
      keywords.append(words[i])
    i += 1

  keywords = [word for word in keywords if word in body]
  return keywords
