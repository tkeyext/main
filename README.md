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
(2) n-gram value,
(3) tf-idf,
(4) entity value



A **Logistic Regression** model will be trained using the features and results. The results will be valued whether the tokens were annotated as keywords or not.

## Evaluation Method
Around 1000 stories will be selected randomly and it will be exluded from the dataset to be used as a test for the model. The results of the model will be compared with the real values of the data and thus we will obtain the accuracy score of the model.

# Results
## Data Evaluation
The data we used for this model was not of high standard. The annotators were confused whether they were supposed to annottage keywords or tags. We had examples such as a word `X` was keyworded in one document but not in another. These examples neutralized our model. In future, ambiguous keywords will be eliminated.

## Feature Evaluation
The features we used for this model were problematic due to; 
1. entity detection in Turkish is itself another project, 
2. tf-idf calculation requires a database with multiple indexing rather than a csv file. 

For (1), `spacy` package were not suitable for Turkish data. The package detected *burayı* as a location whereas *Porsche* as a non-entity.
For (2), it took hours to extract features of the 10% of data because of the usage of CSVs. We had never been able to work with a full data, we either had to cancel it to change a feature or it was shutdown due to memory issues.

For other features, they were all in negative correlation with minimal values.
## Evaluation Evaluation
The evaluation itself can return misleading scores depending on how the data was sampled. If we try to use a set of 50% percent keyworded words and %50 percent of non-keyworded words, a result of 50% accuracy score means that our model is either exaggerating (everything-is-keyword) or underestimating (nothing-is-keyword). In `model2`, we got an accuracy score of ~80% but when we run it against real texts, we realized it exaggerates also. Tests with different sample sizes are required in future.

## Product Evaluation
Although the tests returned with ~80% accuracy when we use a learning rate greater than 0.01, using a custom extractor featurize all words inside a paragraph, we realized our model extracted almost all of the words as a keyword. Everytime we had to change the model and the features a little bit, we had to wait for hours for the features to be generated.

## Future Plans
1. A gold data (in which all the keywords are **correctly** annotated) with 1000 samples would have been much more helpful than a poorly annotated data with more than 50k samples. This will be prioritized in the future.
2. Instead of relying on packages such as `spacy`, the future project will send requests to Wikipedia to check whether a page exists of the given input, and stores it in a text indexed database.
3. A more dynamic TFIDF base will be prioritized. In this state, we had to check the words against a CSV file (with no index), which took a lot of time. 
4. A new feature for checking the **properness** of words is required. For instance, *Dilara Özercan* is a person according to `spacy` but not *Ayşegül Dilara Özercan*. 
5. The features of capitalization and n-gram will be deprecated.
6. Next step for us to train a data with:
   1. Correctly annotated data instead of conflicting data,
   2. With features of properness, tfidf, and wikipedia entity instead of capitalization, n-grams, tfidf, and spacy entity,
   3. With supporting technologies of no-SQL databases instead of CSVs and DataFrames.