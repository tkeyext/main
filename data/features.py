from wordfreq import word_frequency as wf

def uppercased (text: str) -> int:
  """Returns 1 if text is uppercased at some point."""
  return int(text == text.lower())

def ngram (text: str) -> int:
  """Returns the ngram value of the text"""
  return int(len(text.split()))

def totalFreq (text: str) -> float:
  """Returns the total frequency of the word among all."""
  max = wf('ve', 'tr') # Assing the most common word the value 1
  return float(wf(text, 'tr') / float(max))

