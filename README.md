# Twitter_sentiment_analysis_using_NLP

📌 Project Overview

This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and machine learning. The dataset contains tweets labeled as positive, negative, or neutral, and the goal is to classify new tweets into these categories using a Random Forest model with optimized hyperparameters.

The dataset is sourced from Kaggle and contains tweets with sentiment labels:

Column	Description: 

1.Sentiment- (positive, negative, neutral)

2.Reviews

✅ The dataset has been preprocessed to remove noise, special characters, and stopwords before training.


⚙️ Technologies Used

1.Python
2.NLP Libraries: NLTK,  Scikit-learn

3.Machine Learning: Random Forest

4.Feature Engineering: TF-IDF Vectorization

5.Notebook (Google Colab)

6.Git & GitHub for version control


** Model Training & Optimization

The project goes like initially trained several models, but Random Forest performed the best with 91% accuracy after hyperparameter tuning.

🔹 Hyperparameter Tuning:

(Best Parameters)


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

📊 Test the Model
Use the trained model to predict sentiment for new tweets


# Sample tweets for testing

new_texts = [
    "I never expected things to turn out this way.",
    "That was definitely something different.",
    "I guess this is what I was waiting for."
]




👨‍💻 Author
Sumanta Jyoti (DIP)
