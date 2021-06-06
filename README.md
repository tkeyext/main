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