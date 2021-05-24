from nltk import tokenize
from operator import itemgetter
import math
import pandas as pd
import source
import scores

raw_data = pd.read_csv('./data/raw.csv')

