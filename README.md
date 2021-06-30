# Turkish Keyword Extractor

## Definition of the task
**Turkish Keyword Extractor** is a program that automatically extracts keywords from a text.

A **keyword** is a word or a group of words that can be interpreted as:
- a search term for a text
- a topic or a subtopic of a text 
- a word that occurs in a text more than chance frequency.

## Dataset
Turkish data is gathered from **Aposto!** Public API which includes:
- 14 months of daily newsletter
- 380~ issues
- over 8k tagged stories (these can be either long articles or short summaries of an event).

## Framework
The words are tokenized by filtering out the punctuation marks and stopwords. Each token will then get its assigned **features**;
(1) capitalization,
(2) term frequency,
(3) tf-idf,
(4) term length,
(5) n-gram value,
(6) properness.


A **Logistic Regression** model will be trained using the features and results. The results will be valued whether the tokens were annotated as keywords or not.

## Evaluation Method
Around 1000 stories will be selected randomly and it will be exluded from the dataset to be used as a test for the model. The results of the model will be compared with the real values of the data and thus we will obtain the accuracy score of the model.

# Results

## Data Evaluation
The data we used for this model was not of high standard. The annotators were confused whether they were supposed to annottage keywords or tags. We had examples such as a word `X` was keyworded in one document but not in another. These examples neutralized our model. In future, ambiguous keywords will be eliminated.

## Feature Evaluation

## Evaluation Evaluation
The evaluation itself can return misleading scores depending on how the data was sampled. If we try to use a set of 50% percent keyworded words and %50 percent of non-keyworded words, a result of 50% accuracy score means that our model is either exaggerating (everything-is-keyword) or underestimating (nothing-is-keyword). In `model2`, we got an accuracy score of ~80% but when we run it against real texts, we realized it exaggerates also. Tests with different sample sizes are required in future.