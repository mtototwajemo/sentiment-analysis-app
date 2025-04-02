import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_curve, auc
from langdetect import detect
import io
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Define a consistent color palette
SENTIMENT_COLORS = {'positive': '#66c2a5', 'negative': '#fc8d62', 'neutral': '#8da0cb'}

# Function to clean and process text
def clean_text(text, remove_stopwords=True, exclude_words=None):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if remove_stopwords:
        text = ' '.join(word for word in text.split() if word not in stop_words)
    if exclude_words:
        text = ' '.join(word for word in text.split() if word not in exclude_words)
    return text

# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

# Function to map continuous sentiment scores to categorical labels
def map_to_categorical(value, pos_threshold=0.05, neg_threshold=-0.05):
    try:
        value = float(value)
        if value >= pos_threshold:
            return 'positive'
        elif value <= neg_threshold:
            return 'negative'
        else:
            return 'neutral'
    except (ValueError, TypeError):
        # If the value is already a string (e.g., "positive", "neg"), return it after standardizing
        value = str(value).lower()
        if 'pos' in value:
            return 'positive'
        elif 'neg' in value:
            return 'negative'
        elif 'neu' in value:
            return 'neutral'
        return value  # Return as-is if it doesn't match known categories

# Function to detect columns
def detect_columns(df):
    text_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'description'])]
    sentiment_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sentiment', 'label', 'target', 'class', 'score'])]
    timestamp_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])]
    return text_candidates, sentiment_candidates, timestamp_candidates

# Function to train a sentiment model
@st.cache_resource
def train_sentiment_model():
    # Load a small labeled dataset for training (replace with your own labeled data)
    data = {
        'text': [
            "I love this product, it's amazing and works perfectly",
            "This is the worst purchase I've ever made, terrible quality",
            "The item is okay, nothing special but it works",
            "Fantastic service, I'm so happy with my order",
            "Horrible experience, I will never buy again",
            "It's decent, does the job but could be better",
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral']
    }
    df_train = pd.DataFrame(data)
    df_train['cleaned_text'] = df_train['text'].apply(lambda x: clean_text(x, remove_stopwords=True))

    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df_train['cleaned_text'])
    y = df_train['sentiment']

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save the model and vectorizer
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    return model, vectorizer

# Function to load the model and vectorizer
def load_model_and_vectorizer():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        logging.info("Training new model as saved model not found.")
        model, vectorizer = train_sentiment_model()
    return model, vectorizer

# Function to predict sentiment
def predict_sentiment(texts, model, vectorizer, pos_threshold=0.5, neg_threshold=0.5):
    cleaned_texts = [clean_text(text) for text in texts]
    X = vectorizer.transform(cleaned_texts)
    probs = model.predict_proba(X)
    predictions = []
    for prob in probs:
        # Probabilities are ordered as [negative, neutral, positive]
        neg_prob, neu_prob, pos_prob = prob
        if pos_prob >= pos_threshold:
            predictions.append('positive')
        elif neg_prob >= neg_threshold:
            predictions.append('negative')
        else:
            predictions.append('neutral')
    return predictions, probs

# Visualization functions
def plot_sentiment_distribution(sentiment_counts):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           colors=[SENTIMENT_COLORS.get(sentiment, '#d3d3d3') for sentiment in sentiment_counts.index],
           startangle=90, wedgeprops={'edgecolor': 'white'})
    ax.set_title("Sentiment Distribution", fontsize=14, pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_word_cloud(words, sentiment):
    if words:
        wordcloud = WordCloud(width=400, height=300, background_color='white',
                              stopwords=stop_words).generate(words)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{sentiment.capitalize()} Words', fontsize=12)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

def plot_sentiment_trend(df, timestamp_column):
    try:
        df['timestamp'] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        trend = df.groupby([df['timestamp'].dt.date, 'sentiment']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        for sentiment in trend.columns:
            ax.plot(trend.index, trend[sentiment], label=sentiment, color=SENTIMENT_COLORS.get(sentiment))
        ax.set_title("Sentiment Trend Over Time", fontsize=14, pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Reviews", fontsize=12)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not process timestamp column: {e}")

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    ax.set_title("Confusion Matrix", fontsize=14, pad=20)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_roc_curve(y_true, y_prob, classes):
    if len(classes) == 2:
        y_true_binary = np.where(y_true == classes[1], 1, 0)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

def get_top_features(model, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_
    if len(model.classes_) == 2:  # Binary classification
        top_positive = np.argsort(coef[0])[-n:]
        top_negative = np.argsort(coef[0])[:n]
        return [(feature_names[i], coef[0][i]) for i in top_positive], [(feature_names[i], coef[0][i]) for i in top_negative]
    else:  # Multi-class
        top_features = {}
        for i, class_label in enumerate(model.classes_):
            top_indices = np.argsort(coef[i])[-n:]
            top_features[class_label] = [(feature_names[idx], coef[i][idx]) for idx in top_indices]
        return top_features

def main():
    st.title("Sentiment Analysis Dashboard")
    st.markdown("""
    Upload your dataset (CSV or TXT) to analyze the sentiment of text data. The app trains a machine learning model to classify sentiments,
    provides interactive visualizations, and allows you to download the processed dataset with sentiment labels.
    """)

    # Load or train the model
    model, vectorizer = load_model_and_vectorizer()

    # File Upload Section
    st.header("1. Upload Data 📂")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or TXT)", type=["csv", "txt"])

    if uploaded_file:
        # Read the Uploaded File
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, delimiter='\t', encoding='utf-8')
        except Exception as e:
            st.error(f"Error loading file: {e}")
            logging.error(f"Error loading file: {e}")
            return

        st.write("**Dataset Preview:**")
        st.write(df.head())

        # Auto-detect columns
        text_cols, sentiment_cols, timestamp_cols = detect_columns(df)
        default_text = text_cols[0] if text_cols else df.columns[0]
        default_timestamp = timestamp_cols[0] if timestamp_cols else None

        # Column selection
        review_column = st.selectbox("Select text column:", df.columns, index=df.columns.get_loc(default_text))
        sentiment_column = st.selectbox("Select sentiment column (optional, for evaluation):", [None] + list(df.columns),
                                        index=df.columns.get_loc(sentiment_cols[0]) if sentiment_cols else 0)
        timestamp_column = st.selectbox("Select timestamp column (optional):", [None] + list(df.columns),
                                       index=df.columns.get_loc(default_timestamp) if default_timestamp else 0)

        # Data Preprocessing
        st.header("2. Data Preprocessing 🛠️")
        remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        exclude_words = set(list(df.columns) + ['user', 'profile', 'id', 'name'])
        df['cleaned_text'] = df[review_column].apply(lambda x: clean_text(x, remove_stopwords, exclude_words))

        # Show cleaned data preview with toggle
        if st.checkbox("Show Cleaned Data Preview"):
            st.write("**Cleaned Data Preview:**")
            st.write(df[[review_column, 'cleaned_text']].head())

        # Detect language
        sample_text = ' '.join(df['cleaned_text'].head(5))
        language = detect_language(sample_text)
        st.write(f"**Detected Language:** {language.upper()}")

        # Perform Sentiment Analysis
        st.header("3. Sentiment Analysis Results 📊")
        # Custom thresholds for prediction
        st.subheader("Adjust Sentiment Thresholds for Prediction")
        pos_threshold = st.slider("Positive Threshold", 0.0, 1.0, 0.5)
        neg_threshold = st.slider("Negative Threshold", 0.0, 1.0, 0.5)

        predictions, probs = predict_sentiment(df['cleaned_text'], model, vectorizer, pos_threshold, neg_threshold)
        df['sentiment'] = predictions
        df['sentiment_probs'] = [dict(zip(model.classes_, prob)) for prob in probs]

        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
        st.write(sentiment_counts)
        plot_sentiment_distribution(sentiment_counts)

        # Word Clouds for Positive and Negative Sentiments
        st.subheader("Word Clouds")
        col1, col2 = st.columns(2)
        for sentiment, col in [('positive', col1), ('negative', col2)]:
            with col:
                words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                plot_word_cloud(words, sentiment)

        # Time Series Analysis
        if timestamp_column and timestamp_column in df.columns:
            st.subheader("Sentiment Trend Over Time")
            plot_sentiment_trend(df, timestamp_column)

        # Model Evaluation (if labeled data is provided)
        if sentiment_column and sentiment_column in df.columns:
            st.subheader("Model Evaluation")
            y_pred = df['sentiment']
            # Convert y_true to categorical labels if necessary
            y_true = df[sentiment_column].apply(lambda x: map_to_categorical(x, pos_threshold=0.05, neg_threshold=-0.05))

            # Validate that y_true and y_pred are compatible
            unique_true = set(y_true.unique())
            unique_pred = set(y_pred.unique())
            expected_labels = {'positive', 'negative', 'neutral'}
            if not (unique_true.issubset(expected_labels) and unique_pred.issubset(expected_labels)):
                st.error("The sentiment column contains values that cannot be mapped to 'positive', 'negative', or 'neutral'. "
                         "Please ensure the column contains either categorical labels (e.g., 'positive', 'neg', 'neutral') "
                         "or continuous scores that can be thresholded.")
                return

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

            # Display metrics with explanations
            st.write("### Evaluation Metrics")
            st.write("**Accuracy**: The proportion of correct predictions out of all predictions.")
            st.write("Formula: `Accuracy = (Correct Predictions) / (Total Predictions)`")
            st.write("Useful if classes are balanced (equal positive & negative examples).")
            st.write(f"**Accuracy**: {accuracy:.2f}")

            st.write("**Precision**: Measures how many positive predictions were actually correct.")
            st.write("Formula: `Precision = TP / (TP + FP)`")
            st.write("High precision means few false positives (good when false alarms are costly).")
            st.write(f"**Precision (weighted)**: {precision:.2f}")

            st.write("**Recall (Sensitivity)**: Measures how many actual positives were correctly predicted.")
            st.write("Formula: `Recall = TP / (TP + FN)`")
            st.write("High recall means low false negatives (important if missing positives is bad).")
            st.write(f"**Recall (weighted)**: {recall:.2f}")

            st.write("**F1-Score**: A balance between precision and recall.")
            st.write("Formula: `F1-Score = 2 × (Precision × Recall) / (Precision + Recall)`")
            st.write("Good for imbalanced datasets.")
            st.write(f"**F1-Score (weighted)**: {f1:.2f}")

            # Detailed Classification Report
            st.write("**Classification Report**: Detailed metrics for each class.")
            st.text(classification_report(y_true, y_pred))

            # Confusion Matrix
            st.write("**Confusion Matrix**: Shows how many predictions were correct/incorrect for each class.")
            plot_confusion_matrix(y_true, y_pred, model.classes_)

            # ROC Curve (for binary classification)
            if len(model.classes_) == 2:
                st.write("**ROC Curve**: Visualizes the trade-off between true positive rate and false positive rate.")
                plot_roc_curve(y_true, probs, model.classes_)

        # Feature Importance
        st.subheader("Feature Importance (Top TF-IDF Features)")
        if len(model.classes_) == 2:
            top_positive, top_negative = get_top_features(model, vectorizer)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Positive Features:**")
                st.write(pd.DataFrame(top_positive, columns=['Feature', 'Coefficient']))
            with col2:
                st.write("**Top Negative Features:**")
                st.write(pd.DataFrame(top_negative, columns=['Feature', 'Coefficient']))
        else:
            top_features = get_top_features(model, vectorizer)
            for sentiment, features in top_features.items():
                st.write(f"**Top Features for {sentiment.capitalize()}:**")
                st.write(pd.DataFrame(features, columns=['Feature', 'Coefficient']))

        # Interactive Filter
        st.subheader("Filter Reviews by Sentiment")
        selected_sentiment = st.selectbox("Select sentiment to filter:", ['All'] + list(df['sentiment'].unique()))
        filtered_df = df[df['sentiment'] == selected_sentiment] if selected_sentiment != 'All' else df
        st.write(f"**Filtered Reviews ({selected_sentiment}):**")
        st.write(filtered_df[[review_column, 'sentiment']].head())

        # Downloadable Report
        st.header("4. Download Processed Data 📥")
        output = io.BytesIO()
        df_to_download = df[[review_column, 'cleaned_text', 'sentiment']]
        if 'timestamp' in df.columns:
            df_to_download['timestamp'] = df['timestamp']
        df_to_download.to_csv(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download Processed Data",
            data=output,
            file_name="processed_sentiment_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()