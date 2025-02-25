# Twitter_sentiment_analysis_using_NLP

📌 Project Overview
This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and machine learning. The dataset contains tweets labeled as positive, negative, or neutral, and the goal is to classify new tweets into these categories using a Random Forest model with optimized hyperparameters.

The dataset is sourced from Kaggle and contains tweets with sentiment labels:

Column	Description
sentiment Sentiment label (positive, negative, neutral)
review	Tweet text content

✅ The dataset has been preprocessed to remove noise, special characters, and stopwords before training.

⚙️ Technologies Used
Python
NLP Libraries: NLTK, Scikit-learn
Machine Learning: Random Forest
Feature Engineering: TF-IDF Vectorization
Jupyter Notebook (Google Colab)
Git & GitHub for version control

** Model Training & Optimization
The project initially trained several models, but Random Forest performed the best with 91% accuracy after hyperparameter tuning.

🔹 Hyperparameter Tuning (Best Parameters)
python
Copy
Edit
{
    'n_estimators': 300,
    'max_depth': None,
    'max_features': 'log2',
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
🔹 Final Model Accuracy: 91%


📌 How to Run the Project

1️⃣ Clone the Repository

git clone https://github.com/dip-d10/Twitter_sentiment_analysis.git


2️⃣ Install Dependencies

3️⃣ Run the Jupyter Notebook

📊 Testing the Model
Use the trained model to predict sentiment for new tweets

# Load the saved model and vectorizer
rf_model = joblib.load("models/sentiment_analysis_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Sample tweets for testing
new_texts = [
    "I never expected things to turn out this way.",
    "That was definitely something different.",
    "I guess this is what I was waiting for."
]

# Transform and predict
new_texts_transformed = tfidf_vectorizer.transform(new_texts)
predictions = rf_model.predict(new_texts_transformed)

# Output results
for text, sentiment in zip(new_texts, predictions):
    print(f"Text: \"{text}\" --> Sentiment: {sentiment}")

👨‍💻 Author
Sumanta Jyoti (DIP)
