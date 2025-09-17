import streamlit as st
import joblib

# add title to streamlit webapp
st.title('Movie Review Sentiment Analyzer')

# add some text to describe the application
st.subheader('This app is designed to take in a movie review and classify the sentiment of the movie review as either being Positive or Negative.')

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

# if 'Analyze' button is pressed, create and return the sentiment prediction
if st.button('Analyze'):
    # create prediction on the iterable review object using the loaded sentiment model
    pred = sentiment_model.predict(review_list)
    pred_proba = sentiment_model.predict_proba(review_list)

    # change the color of the output based on sentiment being 'positive' (green) or 'negative' (red)
    if pred[0] == 'positive':
        st.subheader(f'Sentiment is :green[{pred[0]}]')
    
    if pred[0] == 'negative':
        st.subheader(f'Review is :red[{pred[0]}]')
    
    # output the prediction probabilities:
    st.subheader('Prediction Probabilities:')
    st.text(f'Negative probability: {pred_proba[0][0]}')
    st.text(f'Positive probability: {pred_proba[0][1]}')
    




