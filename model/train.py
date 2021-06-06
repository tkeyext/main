# Approximation
# f(w, b) = wx + b
import numpy as np
from scipy.sparse import data

class LogisticRegression:
  def __init__ (self, learning_rate = 0.001, iterations = 1000):
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.weights = None
    self.bias = None

  def fit (self, x: np.ndarray, y):
    # init parameters
    sample_size, feature_size = x.shape
    self.weights = np.zeros(feature_size)
    self.bias = 0

    # gradient descent
    for _ in range(self.iterations):
      linear_model = np.dot(x, self.weights) + self.bias
      # print(linear_model[0])
      y_predicted = self._sigmoid(linear_model)

      weight_derivative = (1 / sample_size) * np.dot(x.T, (y_predicted-y))
      bias_derivative = (1 / sample_size) * np.sum(y_predicted - y)

      self.weights -= self.learning_rate * weight_derivative
      self.bias -= self.learning_rate * bias_derivative

  def predict (self, x):
    linear_model = np.dot(x, self.weights) + self.bias
    y_predicted = self._sigmoid(linear_model)
    y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_classes

  def _sigmoid (self, x):
    return 1 / (1 + np.exp(-x))

def run (x: np.ndarray, y: np.ndarray) -> list:
  from sklearn.model_selection import train_test_split
  # from sklearn import datasets
  # import matplotlib as plt
  
  # bc = datasets.load_breast_cancer()
  # x, y = bc.data, bc.target
  print('x:', x)
  print('y:', y)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
  # print(type(x_train[0]))
  def accuracy (y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy

  regressor = LogisticRegression(learning_rate=0.0001, iterations=1000)
  regressor.fit(x_train, y_train)
  predictions = regressor.predict(x_test)
  
  print("LR classification accuracy:", accuracy(y_test, predictions))
  return predictions

x = np.ndarray((2,4), buffer=np.array([1,2,3,4]+[1,1,1,1]))
y = np.ndarray((2,1), buffer=np.array([1,0]))
print(x)
run(x, y)