import numpy as np
import json

class LogisticRegression:
  def __init__ (self, learning_rate = 0.001, iterations = 1000):
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.weights = None
    self.bias = None

  def fit (self, x: np.ndarray, y):
    # init parameters
    sample_size, feature_size = x.shape
    # print("\nSample size is:", sample_size, "\nFeature size is:", feature_size)
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

  def predict (self, x, breakpoint = 0.5):
    # print("\n X is:", x, "\nWeights are:", self.weights, "\nBias is:", self.bias)
    linear_model = np.dot(x, self.weights) + self.bias
    y_predicted = self._sigmoid(linear_model)
    # print("\nPredicted is:", y_predicted)
    y_predicted_classes = [1 if i > breakpoint else 0 for i in y_predicted]
    return y_predicted_classes

  def _sigmoid (self, x):
    return 1 / (1 + np.exp(-x))

  def load (self, filename):
    with open(f'./bin/{filename}.txt') as jsonfile:
      data = json.load(jsonfile)
      self.learning_rate = data['learning_rate']
      self.iterations = data["iterations"]
      self.weights = np.array(data["weights"])
      self.bias = data["bias"]
      # print(self.learning_rate, self.iterations, self.weights, self.bias)
      
  def save (self, filename):
    data = {
      'learning_rate': self.learning_rate, 
      'iterations': self.iterations,
      'weights': self.weights.tolist(),
      'bias': self.bias
    }
    with open(f'bin/{filename}.txt', "w") as outfile:
      json.dump(data, outfile)
