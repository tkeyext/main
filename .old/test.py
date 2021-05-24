# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# x = np.arange(10).reshape(-1, 1)
# y = np.array([0,0,0,0,1,1,1,1,1,1])

# model = LogisticRegression(solver='liblinear', random_state=0)

# model.fit(x, y)

# # print(model.predict(x))

# print(model.score(x, y))

# confusion_matrix(y, model.predict(x))

import source

source.getWords()