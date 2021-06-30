from numpy import string_
import pandas as pd

def getDataSampled ():
  """Samples the data by their value of keyword."""
  words = pd.read_csv('./data/words.csv').rename_axis('ID')
  n = len(words[words['Keyword'] == 1])
  df_1 = words[words['Keyword'] == 1]
  df_0 = words[words['Keyword'] == 0][0:n*5]

  sampled = pd.concat([df_0, df_1], ignore_index=True).rename_axis('ID')
  print(sampled)
  sampled.to_csv('./data/sampled.csv')

# getDataSampled()