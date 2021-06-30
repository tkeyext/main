# Approximation
# f(w, b) = wx + b
import numpy as np
import pandas as pd
from .regression import LogisticRegression


def run (x: np.ndarray, y: np.ndarray) -> list:
  from sklearn.model_selection import train_test_split
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1234)
  
  regressor = LogisticRegression(learning_rate=0.01, iterations=1000)
  
  regressor.fit(x_train, y_train)

  predictions = regressor.predict(x_test)

  regressor.save('model2')
  
  def accuracy (y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy
  
  print("LR classification accuracy:", accuracy(y_test, predictions))
  return predictions

def start(path: str):
  training_data = pd.read_csv(path)
  x = training_data['X'].apply(lambda x: eval(x), 0).tolist()
  X = pd.DataFrame(
      x,
      columns=['a', 'b', 'c', 'd']
    ).to_numpy()
  y = training_data['Y'].astype('int').to_numpy()
  run(X, y)