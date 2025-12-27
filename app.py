import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Sentiment Tracker",
    page_icon="📊",
    layout="centered"
)

# ---------------- DARK UI STYLE ----------------
st.markdown("""
<style>
.stProgress > div > div > div > div {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Sentiment Tracker")
st.write("Analyze customer reviews in real-time using Machine Learning")

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=True)
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

try:
    model, vectorizer = load_model()
except:
    st.error("❌ Model files not found. Please check your repository.")
    st.stop()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- INPUT ----------------
review = st.text_area(
    "✍️ Enter customer review:",
    height=150,
    placeholder="Example: The service was excellent and delivery was fast!"
)

# ---------------- LIVE PREDICTION ----------------
if review.strip():

    review_vec = vectorizer.transform([review])
    prob = model.predict_proba(review_vec)[0][1]

    # Confidence logic
    if prob >= 0.75 or prob <= 0.25:
        confidence = "High"
        color = "🟢"
    elif prob >= 0.6 or prob <= 0.4:
        confidence = "Medium"
        color = "🟡"
    else:
        confidence = "Low"
        color = "🔴"

    st.progress(int(prob * 100))

    if prob >= 0.5:
        st.success(f"😊 Positive Review — {prob*100:.2f}%")
    else:
        st.error(f"☹️ Negative Review — {(1-prob)*100:.2f}%")

    st.markdown(f"### {color} Confidence Level: **{confidence}**")

    # Save history
    st.session_state.history.append(prob)

    # ---------------- TF-IDF WORD CLOUD ----------------
    st.subheader("☁️ Influential Words")

    feature_names = vectorizer.get_feature_names_out()
    weights = review_vec.toarray()[0]

    word_freq = {
        feature_names[i]: weights[i]
        for i in range(len(weights))
        if weights[i] > 0
    }

    if word_freq:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="black"
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

# ---------------- CHART ----------------
if len(st.session_state.history) > 1:
    st.subheader("📈 Sentiment Probability Over Time")

    plt.figure()
    plt.plot(st.session_state.history)
    plt.xlabel("Review Count")
    plt.ylabel("Positive Probability")
    st.pyplot(plt)

# ---------------- RESET ----------------
if st.button("♻️ Reset"):
    st.session_state.history = []
    st.rerun()

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, TF-IDF & Machine Learning")
