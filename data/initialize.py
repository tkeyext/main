from pandas.io import json
import requests
import math
import pandas as pd
import sqlite3
import re

BASE_URL = 'https://api.apos.to'
PAGE = 0

def getStoryCount ():
  response = requests.get(f'{BASE_URL}/stories/list?limit=1&newsletter={getNewsletterId()}')
  return response.json()['count']

def storeStory (story = None):
  raw_data = pd.read_csv('./data/raw.csv')
  raw_data.head()

def getNewsletterId ():
  response = requests.get(f'{BASE_URL}/newsletters/daily')
  return response.json()['data']['_id']

def getStories ():
  count = getStoryCount()
  pages = range(math.floor(count / 1000))
  df = pd.DataFrame(columns=('ID', 'Body', 'Tags'))
  print(df)
  for page in pages:
    print(page)
    request = f'{BASE_URL}/stories/list?sort=publishedAt&desc=true&limit=1000&page={page}&newsletter={getNewsletterId()}&type=bullet'
    stories = requests.get(request)
    stories = list(stories.json()['data'])
    for story in stories:
      if len(story['tags']) > 0:

        ID = story['_id']
        
        Body = story['body']['text']
        Body = re.sub(',', '', Body)

        Tags = '??'.join(list(story['tags']))
        df = df.append({ 'ID': ID, 'Body': Body, 'Tags': Tags }, ignore_index=True)
    print(df)
  
  df.to_csv('./data/raw.csv')
  
def getWords ():
  df = pd.DataFrame(columns=['Word', 'Tag'])
  stories = pd.read_csv('./data/raw.csv')
  for i, row in stories.iterrows():
    print('Processing row', i)
    Body = str(row['Body']).strip()
    Tags = str(row['Tags']).split('??')
    Body = re.sub("[^\w]+", " ", str(Body))
    Words = Body.split(' ')
    for tag in Tags:
      for word in Words:
        print(word, '-', tag)
        df.append({ 'Word': word, 'Tag': tag }, ignore_index=True)
  df.to_csv('./data/processed.csv')

getStories()