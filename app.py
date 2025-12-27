import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title("Customer Sentiment Tracker (Live)")

# Input box
review = st.text_area("Type your review here:", height=150)

# Only predict if there's input
if review:
    # Vectorize input
    review_vec = vectorizer.transform([review])
    
    # Predict probability
    prob = model.predict_proba(review_vec)[0][1]  # Probability of positive
    
    # Display sentiment bar
    st.write("Sentiment Probability:")
    st.progress(int(prob * 100))
    
    # Color indicator
    if prob > 0.5:
        st.success(f"Positive Review ({prob*100:.2f}%)")
    else:
        st.error(f"Negative Review ({(1-prob)*100:.2f}%)")
