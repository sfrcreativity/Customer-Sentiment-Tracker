import os
import pickle
import streamlit as st

model_path = 'sentiment_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
else:
    st.error("Model or vectorizer file not found. Please check the path!")


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
