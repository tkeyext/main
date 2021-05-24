from os import name
import re
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('./data/raw.csv')
docs = data.Body.tolist()
# print(docs)

stopwords = nltk.corpus.stopwords.words('turkish')

def norm_doc(single_doc):
  # TR: Dokümandan belirlenen özel karakterleri ve sayıları at
  # EN: Remove special characters and numbers
  # single_doc = re.sub(" \d+", " ", single_doc)
  pattern = r"[{}]".format(",.;•") 
  single_doc = re.sub(pattern, "", str(single_doc))
  single_doc = re.sub("\[http.+\]", "", str(single_doc))
  # TR: Dokümanı küçük harflere çevir
  # EN: Convert document to lowercase
  single_doc = single_doc.lower()
  single_doc = single_doc.strip()
  # TR: Dokümanı token'larına ayır
  # EN: Tokenize documents
  tokens = nltk.WordPunctTokenizer().tokenize(single_doc)
  # TR: Stop-word listesindeki kelimeler hariç al
  # EN: Filter out the stop-words 
  filtered_tokens = [token for token in tokens if token not in stopwords]
  # TR: Dokümanı tekrar oluştur
  # EN: Reconstruct the document
  single_doc = ' '.join(filtered_tokens)
  return single_doc

norm_docs = np.vectorize(norm_doc) #like magic :)
normalized_documents = norm_docs(docs)
# print(normalized_documents)

BoW_Vector = CountVectorizer(min_df = 0., max_df = 1.)
BoW_Matrix = BoW_Vector.fit_transform(normalized_documents)

features = BoW_Vector.get_feature_names()
print ("features[50]:" + features[46])
print ("features[52]:" +features[48])

BoW_Matrix = BoW_Matrix.toarray()
print(BoW_Matrix)
# TR: Doküman - öznitelik matrisini göster
# EN: Print document by term matrice
BoW_df = pd.DataFrame(BoW_Matrix, columns = features)
BoW_df
#print(BoW_df.info())

# TR: 2.TFxIdf Hesaplama Adımları
# EN: 2.TFxIdf Calculation Steps
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_Vector = TfidfVectorizer(min_df = 0., max_df = 1., use_idf = True)
Tfidf_Matrix = Tfidf_Vector.fit_transform(normalized_documents)
Tfidf_Matrix = Tfidf_Matrix.toarray()
print(np.round(Tfidf_Matrix, 3))
# TR: Tfidf_Vector içerisindeki tüm öznitelikleri al
# EN: Fetch al features in Tfidf_Vector
features = Tfidf_Vector.get_feature_names()
# TR: Doküman - öznitelik matrisini göster
# EN: Print document by term matrice
Tfidf_df = pd.DataFrame(np.round(Tfidf_Matrix, 3), columns = features)
Tfidf_df

from sklearn.decomposition import LatentDirichletAllocation
number_of_topics = 4
BoW_Matrix = BoW_Vector.fit_transform(normalized_documents)
LDA = LatentDirichletAllocation(n_components = number_of_topics, 
                                max_iter = 10, 
                                learning_offset = 50.,
                                random_state = 0,
                                learning_method = 'online').fit(BoW_Matrix)
features = BoW_Vector.get_feature_names()
for t_id, topic in enumerate(LDA.components_):
    print ("Topic %d:" % (t_id))
    print (" ".join([features[i]
                    for i in topic.argsort()[:-number_of_topics - 1:-1]]))

