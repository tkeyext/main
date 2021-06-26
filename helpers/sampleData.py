import pandas as pd

def getDataSampled ():
  """Samples the data by their value of keyword."""
  words = pd.read_csv('./data/words.csv')
  
  df_1 = words[words['Keyword'] == 1].sample(n=500)
  df_0 = words[words['Keyword'] == 0].sample(n=2000)

  sampled = pd.concat([df_0, df_1], ignore_index=True)
  print(sampled)
  sampled.to_csv('./data/sampled.csv')

getDataSampled()