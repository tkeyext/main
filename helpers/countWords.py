import pandas as pd

def countWordOccurences ():
  """Writes a csv file that contains the number of total occurences of words."""
  words = pd.read_csv('./data/words.csv')
  # counts = words["Word"].value_counts().rename_axis('Word')
  counts = words.stack().value_counts().to_frame('n').rename_axis('Word').loc[words.Word]
  counts.to_csv('./data/counts.csv')

countWordOccurences()