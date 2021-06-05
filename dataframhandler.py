import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import timeit
from pandas import DataFrame


def removePunctuations(text: str) -> str:
  """Strips the input from non-word and non-digit characters."""
  text = re.sub('[^\w\d ]', '', text)
  return text

def removeStopwords(text: str) -> str:
  """Stripts the input from stopwords."""
  stop_words = stopwords.words('turkish')

  w = text.split()
  words = []
  for item in w:
    if type(item) == "str":
        item = item.lower()
    words.append(item)

  filtered = [word for word in words if word not in stop_words]
  return ' '.join(filtered)


# def getWords1 ():
#     """processes the raw data to as n-grams"""
#     df = pd.DataFrame(columns=['Word', 'Tag'])
#     stories = pd.read_csv('C:\\Users\kaan3\PycharmProjects\\484\main\.old\\data\\raw.csv')
#     for i, row in stories.iterrows():
#       print('Processing row', i)
#       Body = str(row['Body']).strip()
#       Tags = str(row['Tags']).split('??')
#       Body = re.sub("[^\w]+", " ", str(Body))
#       Words = Body.split(' ')
#       for tag in Tags:
#         for word in Words:
#           print(word, '-', tag)
#           df.append({ 'Word': word, 'Tag': tag }, ignore_index=True)
#     df.to_csv('C:\\Users\kaan3\PycharmProjects\\484\main\.old\\data\\processed.csv')

def getWords():
    """processes the raw data to as n-grams"""
    nltk.download('stopwords')
    start = timeit.timeit()
    print(start)
    df = pd.DataFrame(columns=['Word', 'Tags']) #creating a new dataframe to insert the processed items
    stories = pd.read_csv('./data/raw.csv') #opening the imported stories from aposto!
    print(stories)
    for i in stories.index: #looping over each index to insert the each word into the df
        body = stories["Body"].iloc[i] #getting each row i.e. story
        body = removeStopwords(body) #removing stopwords from the row
        body = removePunctuations(body) #removing punctuations from the row
        body = body.split() #splitting the string to make be able to iterate over
        tag = stories["Tags"].iloc[i] #getting the keywords of our story
        tag = tag.replace("??", " ") #processing the keywords, this is here due to a complication with the API

        for item in body: #iteration over each word in a row
            df2 = pd.Series({"Word": item, "Tags": tag}, index=df.columns) #creating a temporary row to insert it into our dataframe
            df = df.append(
                df2, ignore_index=True
            ) # inserting the row to our dataframe
    df.to_csv('C:\\Users\kaan3\PycharmProjects\\484\main\.old\\data\\processed.csv') #importing the dataframe to csv
    end = timeit.timeit()
    print(end - start)


getWords()

