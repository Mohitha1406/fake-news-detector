import streamlit as st
import joblib
import re
import nltk
import numpy as np

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def simple_summary(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:num_sentences]) if len(sentences) >= num_sentences else text

def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    # Decision function score as confidence proxy
    score = model.decision_function(vec)[0]
    confidence = round(min(100, 50 + abs(score) * 10), 1)
    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence

# ─── UI ───────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake News Detector for Students")
st.markdown("Paste a news article below and find out if it's **REAL** or **FAKE** instantly.")
st.markdown("---")

model, vectorizer = load_model()

article = st.text_area("📰 Paste News Article Here", height=250,
                        placeholder="Copy and paste the full news article text here...")

if st.button("🚀 Analyze Article", use_container_width=True):
    if len(article.strip()) < 50:
        st.warning("Please enter at least 50 characters for a meaningful prediction.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict(article, model, vectorizer)
            summary = simple_summary(article)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if label == "REAL":
                st.success(f"✅ **{label} NEWS**")
            else:
                st.error(f"❌ **{label} NEWS**")

        with col2:
            st.metric("Confidence Score", f"{confidence}%")

        st.markdown("### 📝 Article Summary")
        st.info(summary)

        st.markdown("---")
        st.caption("⚠️ This tool uses ML and may not be 100% accurate. Always verify from multiple sources.")

st.markdown("---")
st.markdown("Built with ❤️ using Python, Scikit-learn & Streamlit")
