from sklearn.linear_model import LogisticRegression
import pandas as pd

training_data = pd.read_csv("./data/training.csv")

X = pd.DataFrame(training_data['X'].apply(lambda x: eval(x), 0).tolist(), columns=['a', 'b', 'c', 'd']).to_numpy()
y = training_data['Y'].astype('int').to_numpy()

clf = LogisticRegression(random_state=0).fit(X, y)
print(clf)
print(clf.predict(X[:2, :]))

print(clf.predict_proba(X[:2, :]))

print(clf.score(X, y))