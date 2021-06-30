import requests
import math
import pandas as pd
import re

BASE_URL = 'https://api.apos.to'
PAGE = 0

def getStoryCount ():
  """Fetches the total story count so that we can paginate our requests."""
  response = requests.get(f'{BASE_URL}/stories/list?limit=1&newsletter={getNewsletterId()}')
  return response.json()['count']

def getNewsletterId ():
  """Fetches the target newsletter's id, so that we can pick stories from only one newsletter."""
  response = requests.get(f'{BASE_URL}/newsletters/daily')
  return response.json()['data']['_id']

def getStories ():
  """Fetches the stories and maps their values to a dataframe."""
  count = getStoryCount()
  pages = range(math.floor(count / 999))
  df = pd.DataFrame(columns=('ID', 'Body', 'Tags'))
  for page in pages:
    request = f'{BASE_URL}/stories/list?sort=publishedAt&desc=true&limit=999&page={page}&newsletter={getNewsletterId()}&type=bullet'
    stories = requests.get(request)
    stories = list(stories.json()['data'])
    for story in stories:
      if len(story['tags']) > 0:

        ID = story['_id']
        
        Body = story['body']['text']
        Body = re.sub(',', '', Body)

        Tags = '??'.join(list(story['tags']))
        df = df.append({ 'ID': ID, 'Body': Body, 'Tags': Tags }, ignore_index=True)
  
  df.to_csv('./data/documents.csv')

# getStories()