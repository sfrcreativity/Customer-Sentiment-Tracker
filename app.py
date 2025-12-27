import streamlit as st
import pickle
import os

st.title("Customer Sentiment Tracker (Live)")

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

# Cache the model & vectorizer to load only once
@st.cache_data(show_spinner=True)
def load_model(model_path, vectorizer_path):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or vectorizer file not found. Please upload them to the repo.")
        return None, None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)

# Stop if model failed to load
if model is None or vectorizer is None:
    st.error("Model or vectorizer file not found. Please upload them to the repo.")
    st.stop()

# Text input
review = st.text_area("Type your review here:", height=150)

if review:
    # Transform input
    review_vec = vectorizer.transform([review])
    
    # Predict probability
    prob = model.predict_proba(review_vec)[0][1]  # Probability of positive
    
    # Display progress bar
    st.progress(int(prob * 100))
    
    # Display sentiment with color
    if prob > 0.5:
        st.success(f"Positive Review ({prob*100:.2f}%)")
    else:
        st.error(f"Negative Review ({(1-prob)*100:.2f}%)")
