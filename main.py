#imports

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import re

#load models
model = pickle.load(open('model_emotion.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

#custom functions
stopwords = nltk.corpus.stopwords.words('english')


def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if not word in set(stopwords)]
    return " ".join(text)


def pred_emotion(text):
    # Clean the input text and ensure it is in a list form
    cleaned_text = clean_text(text)  # Should return a list like ['cleaned text']

    # Transform the cleaned text using the TF-IDF vectorizer
    input_vectorized = tfidf.transform([cleaned_text])

    # Predict the label using the logistic regression model
    predicted_label = model.predict(input_vectorized)[0]  # Get the first (and only) prediction

    # Inverse transform the label to get the emotion name
    emotion = lb.inverse_transform([predicted_label])[0]

    prob = np.max(model.predict(input_vectorized))

    # Return the predicted emotion and label
    return emotion, predicted_label, prob


#app
st.title("Six NLP Emotions Detection App")
st.write(['Joy', 'Fear', 'Love', 'Anger', 'Sadness', 'Surprise'])
input_text = st.text_input("Paste Your Text Here")

if st.button("predict"):
    predicted_emotion, label, prob = pred_emotion(input_text)
    st.write("Predicted Emotion: ", predicted_emotion)
    st.write("Predicted Label: ", label)
