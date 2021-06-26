import pandas as pd

def getDataSampled ():
  """Samples the data by their value of keyword."""
  words = pd.read_csv('./data/words.csv')
  n = len(words[words['Keyword'] == 1])

  df_1 = words[words['Keyword'] == 1].sample(n)
  df_0 = words[words['Keyword'] == 0].sample(n)

  sampled = pd.concat([df_0, df_1], ignore_index=True)
  print(sampled)
  sampled.to_csv('./data/sampled.csv')

getDataSampled()