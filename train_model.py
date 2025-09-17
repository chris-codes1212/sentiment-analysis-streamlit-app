import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

import joblib

# Read in data as pandas data frame
df = pd.read_csv('IMDB_Dataset.csv')

# create X and y data sets for the model
X = df['review']
y = df['sentiment']

# create model pipeline
# input is a list of tuples containing the name of the estimator, and the estimator type
# all estimators in the pipeline, except for the last one must also contain a transform operation (must be a transformer)
# here there is one transformer (TfidfVectorizer) to create the matrix of TF-IDF features
# the final estimator is the MultinomnialNB (Multinomial Naybe Bayes Classifier) Model
model_pipeline = Pipeline([('tfidf', TfidfVectorizer()), 
                 ('nayve_bayes_clf', MultinomialNB())])

# calling 'fit' on the pipeline is equivalent to running fit and transform on all estimators before the last, which will be the model we are training
# here calling .fit will apply fit and transform with the tfidf vectorizer on the X data to create the tfidf matrix, then will apply fit using the 
# Multinomial Nayve Bayes Classifier on the data set (X and y) to train the model
model_pipeline.fit(X, y)

# now we save the model that we just trained to a file for access by our application in the app.py file
joblib.dump(model_pipeline, 'sentiment_model.pkl')

print("Model pipeline saved to sentiment_model.pkl")

