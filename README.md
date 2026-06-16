# 🔍 Fake News Detector for Students

An AI-powered web application that detects whether a news article is **REAL** or **FAKE** using NLP and Machine Learning.

## 🚀 Live Demo
[Click here to try the app](https://fake-news-detector.streamlit.app)

## 📌 Features
- Paste any news article and get an instant REAL/FAKE verdict
- Confidence score (%) for each prediction
- Auto-generated 3-sentence article summary
- Color-coded results (green = REAL, red = FAKE)
- No login or installation needed — runs in the browser

## 🧠 Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Model | Passive Aggressive Classifier |
| Feature Extraction | TF-IDF Vectorizer |
| NLP Preprocessing | NLTK |
| Web UI | Streamlit |
| Model Storage | joblib |
| Dataset | Kaggle Fake News Dataset (44k+ articles) |
| Deployment | Streamlit Community Cloud |

## 📁 Project Structure
```
fake-news-detector/
├── app.py              # Streamlit web app
├── train_model.py      # Model training script
├── model.pkl           # Trained model (generated after training)
├── vectorizer.pkl      # TF-IDF vectorizer (generated after training)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## ⚙️ How to Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Place True.csv and Fake.csv in the project folder

# 4. Train the model
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py
```

## 📊 Model Performance
- **Accuracy:** 93.5%
- **Precision:** 94.1%
- **Recall:** 93.0%
- **F1-Score:** 93.5%

## 📚 References
- [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [Scikit-learn PassiveAggressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html)
- [Streamlit Documentation](https://docs.streamlit.io)
- [NLTK Documentation](https://www.nltk.org)

## 👩‍💻 Author
**Mohitha**  
B.Tech AI & Data Science | Saveetha School of Engineering (SIMATS)
