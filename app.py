import streamlit as st
import joblib

# add title to streamlit webapp
st.title('Movie Review Sentiment Analyzer')

# add some text to describe the application
st.text('This app is designed to take in a movie review and classify the sentiment of the movie review as either being Positive or Negative.')

# this function will return the trained sentiment model
@st.cache_data
def get_model():
    model = joblib.load('sentiment_model.pkl')
    return model

# get the sentiment model
sentiment_model = get_model()

# get the review in streamlit app from user
review = st.text_area("Review:")
# store the review an iterable object with only the review from the the user
review_list = [review]

# create prediction on the iterable review object using the loaded sentiment model
pred = sentiment_model.predict(review_list)
pred_proba = sentiment_model.predict_proba(review_list)

# if 'Analyze' button is pressed, return the prediction
if st.button('Analyze'):
    pred
    pred_proba

