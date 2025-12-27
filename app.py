import os
import pickle
import streamlit as st

st.title("Customer Sentiment Tracker (Live Dashboard)")

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

# Load and cache model & vectorizer
@st.cache_data(show_spinner=True)
def load_model(model_path, vectorizer_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

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
