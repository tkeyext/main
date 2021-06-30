from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
  

training_data = pd.read_csv("./data/training.csv")

X = pd.DataFrame(training_data['X'].apply(lambda x: eval(x), 0).tolist(), columns=['a', 'b', 'c', 'd']).to_numpy()
y = training_data['Y'].astype('int').to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

clf = LogisticRegression(random_state=0).fit(x_train, y_train)

print(clf.predict(x_test[:2, :]))

print(clf.predict_proba(x_test[:2, :]))

print(clf.score(x_test, y_test))