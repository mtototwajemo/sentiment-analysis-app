import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import io
import joblib
import logging
import json
import chardet  # For detecting file encoding

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a flexible color palette
SENTIMENT_COLORS = {'positive': '#66c2a5', 'negative': '#fc8d62', 'neutral': '#8da0cb', 'good': '#66c2a5', 'bad': '#fc8d62'}

def clean_text(text, remove_stopwords=True, lemmatize=True):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def detect_and_map_sentiment(df, sentiment_column):
    """Detect and map sentiment values dynamically."""
    values = df[sentiment_column].dropna().astype(str).str.lower()
    unique_values = set(values)

    try:
        numeric_values = pd.to_numeric(values, errors='coerce')
        if numeric_values.notna().all():
            min_val, max_val = numeric_values.min(), numeric_values.max()
            if min_val >= 1 and max_val <= 5:
                st.write("Detected 1-5 rating scale. Mapping: 1-2 = negative, 3 = neutral, 4-5 = positive")
                return lambda x: 'negative' if pd.to_numeric(x) <= 2 else ('neutral' if pd.to_numeric(x) == 3 else 'positive')
            elif min_val >= 0 and max_val <= 1:
                st.write("Detected 0-1 score. Mapping: <0.4 = negative, 0.4-0.6 = neutral, >0.6 = positive")
                return lambda x: 'negative' if pd.to_numeric(x) < 0.4 else ('neutral' if pd.to_numeric(x) <= 0.6 else 'positive')
            else:
                st.write("Detected numeric values. Mapping based on terciles.")
                terciles = np.percentile(numeric_values, [33, 66])
                return lambda x: 'negative' if pd.to_numeric(x) <= terciles[0] else ('neutral' if pd.to_numeric(x) <= terciles[1] else 'positive')
    except:
        pass

    common_labels = {'positive', 'negative', 'neutral', 'pos', 'neg', 'neu', 'good', 'bad'}
    if unique_values.issubset(common_labels):
        st.write(f"Detected categorical labels: {unique_values}")
        return lambda x: 'positive' if str(x).lower() in ['positive', 'pos', 'good'] else ('negative' if str(x).lower() in ['negative', 'neg', 'bad'] else 'neutral')

    if len(unique_values) <= 5:
        st.write(f"Detected custom categorical labels: {unique_values}. Using as-is.")
        return lambda x: str(x).lower()

    st.error(f"Unable to interpret '{sentiment_column}' with values: {unique_values}. Please ensure it contains ratings, scores, or clear sentiment labels.")
    return None

def detect_columns(df):
    """Auto-detect text and sentiment columns."""
    text_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'description', 'news', 'tweet'])]
    sentiment_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sentiment', 'label', 'target', 'class', 'score', 'rating'])]

    text_column = text_candidates[0] if text_candidates else [col for col in df.columns if df[col].dtype == 'object'][0]
    sentiment_column = sentiment_candidates[0] if sentiment_candidates else None
    return text_column, sentiment_column

def detect_encoding(file):
    """Detect file encoding."""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer
    return result['encoding']

def load_file(uploaded_file):
    """Load file with encoding detection."""
    try:
        if uploaded_file.name.endswith('.csv'):
            encoding = detect_encoding(uploaded_file)
            return pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            encoding = detect_encoding(uploaded_file)
            return pd.read_csv(uploaded_file, delimiter='\t', encoding=encoding, on_bad_lines='skip')
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            return pd.DataFrame(data)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, TXT, or JSON.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error loading file: {e}")
        return None

@st.cache_resource
def train_sentiment_model(df_train, text_column, sentiment_column, _mapping_func=None):
    """Train a sentiment model with a default dataset if no labels are provided."""
    if sentiment_column is None or _mapping_func is None:
        st.write("No labeled data provided. Training on default Sentiment140 dataset (sample).")
        try:
            default_data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitt30k.csv', encoding='latin-1')
            default_data.columns = ['text', 'sentiment']
            default_data['sentiment'] = default_data['sentiment'].map({0: 'negative', 4: 'positive'}).fillna('neutral')
            df_train = default_data.sample(1000, random_state=42)
            text_column, sentiment_column = 'text', 'sentiment'
            _mapping_func = lambda x: x
        except:
            st.error("Failed to load default dataset. Please provide labeled data.")
            return None, None, 0, 0, 0, 0

    df_train = df_train.dropna(subset=[text_column])  # Remove nulls in text column
    df_train['cleaned_text'] = df_train[text_column].apply(lambda x: clean_text(x, remove_stopwords=True, lemmatize=True))
    df_train['sentiment_mapped'] = df_train[sentiment_column].apply(_mapping_func)

    # Check data balance
    st.write("Sentiment Distribution in Training Data:")
    st.write(df_train['sentiment_mapped'].value_counts())

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df_train['cleaned_text'])
    y = df_train['sentiment_mapped']

    model = LogisticRegression(max_iter=1000, class_weight='balanced')  # Use class weights
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    return model, vectorizer, accuracy, precision, recall, f1

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        return None, None
    return model, vectorizer

def predict_sentiment(texts, model, vectorizer, threshold=0.5):  # Lower the threshold
    """Predict sentiment with a neutral zone."""
    cleaned_texts = [clean_text(text, remove_stopwords=True, lemmatize=True) for text in texts]
    X = vectorizer.transform(cleaned_texts)
    probs = model.predict_proba(X)
    predictions = []
    for prob in probs:
        max_prob = max(prob)
        if max_prob < threshold and 'neutral' in model.classes_:
            predictions.append('neutral')
        else:
            predictions.append(model.classes_[np.argmax(prob)])
    return predictions, probs

def plot_sentiment_distribution(df, sentiment_column, mapping_func, title="Sentiment Distribution"):
    """Plot sentiment distribution."""
    df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=df['sentiment_mapped'], ax=ax, palette=SENTIMENT_COLORS, hue=df['sentiment_mapped'], legend=False)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_word_cloud(df, text_column, sentiment_column, sentiment, mapping_func):
    """Plot word cloud for a specific sentiment."""
    df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
    words = ' '.join(df[df['sentiment_mapped'] == sentiment][text_column].dropna())
    if words:
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=stop_words).generate(words)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{sentiment.capitalize()} Words', fontsize=14)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

def plot_text_length_distribution(df, text_column):
    """Plot text length distribution."""
    df['text_length'] = df[text_column].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['text_length'], bins=30, kde=True, ax=ax, color='#8da0cb')
    ax.set_title("Text Length Distribution", fontsize=14)
    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def display_sentiment_counts(df, sentiment_column, mapping_func, title="Sentiment Counts"):
    """Display sentiment counts."""
    df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
    counts = df['sentiment_mapped'].value_counts()
    st.write(f"**{title}:**")
    for sentiment, count in counts.items():
        st.write(f"{sentiment.capitalize()}: {count}")

def main():
    """Main function to run the Streamlit app."""
    st.title("Super Sentiment Analysis Dashboard")
    st.markdown("""
    Analyze sentiments in stock news, tweets, or financial reports. Upload your data, explore it, train a model,
    and predict sentiments with detailed metrics and visualizations. No external LLM APIs—just pure ML power!
    """)

    # Step 1: Upload & Preview Dataset
    st.header("1. Upload & Preview Dataset 📂")
    uploaded_file = st.file_uploader("Upload CSV, Excel, TXT, or JSON (e.g., stock news, tweets)", type=["csv", "xlsx", "txt", "json"])

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is None or df.empty:
            st.error("Failed to load dataset or dataset is empty.")
            return

        st.write("### Data Preview:")
        st.dataframe(df.head())

        # Auto-detect columns
        text_column, sentiment_column = detect_columns(df)
        st.write(f"**Auto-Detected Columns:** Text: {text_column}, Sentiment: {sentiment_column}")

        # Allow manual override
        text_column = st.selectbox("Override Text Column:", df.columns, index=df.columns.get_loc(text_column))
        sentiment_column = st.selectbox("Override Sentiment Column (optional):", [None] + list(df.columns),
                                        index=df.columns.get_loc(sentiment_column) if sentiment_column else 0)

        # Step 2: Exploratory Data Analysis (EDA)
        st.header("2. Exploratory Data Analysis (EDA) 🕵️‍♂️")
        st.write("### Dataset Summary:")
        st.write(df.describe(include="all"))

        st.write("### Missing Values:")
        st.write(df.isnull().sum())

        mapping_func = None
        if sentiment_column:
            mapping_func = detect_and_map_sentiment(df, sentiment_column)
            if mapping_func is None:
                return

            unique_sentiments = df[sentiment_column].unique()
            st.write(f"### Unique Sentiment Values: {list(unique_sentiments)}")
            display_sentiment_counts(df, sentiment_column, mapping_func, "Sentiment Counts Before Prediction")
            plot_sentiment_distribution(df, sentiment_column, mapping_func, "Sentiment Distribution Before Prediction")

            st.write("### Word Clouds:")
            col1, col2 = st.columns(2)
            with col1:
                plot_word_cloud(df, text_column, sentiment_column, 'positive', mapping_func)
            with col2:
                plot_word_cloud(df, text_column, sentiment_column, 'negative', mapping_func)

        st.write("### Text Length Distribution:")
        plot_text_length_distribution(df, text_column)

        # Step 3: Data Cleaning & Preprocessing
        st.header("3. Data Cleaning & Preprocessing 🛠️")
        remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        lemmatize = st.checkbox("Lemmatize Words", value=True)
        df['cleaned_text'] = df[text_column].apply(lambda x: clean_text(x, remove_stopwords, lemmatize))

        if st.checkbox("Show Cleaned Data Preview"):
            st.write("**Cleaned Data Preview:**")
            st.write(df[[text_column, 'cleaned_text']].head())

        # Step 4: Train & Evaluate a Sentiment Model
        st.header("4. Train & Evaluate Sentiment Model 📈")
        model, vectorizer = load_model_and_vectorizer()

        if st.button("Train Model"):
            if sentiment_column and mapping_func:
                if df[sentiment_column].isnull().sum() > 0:
                    st.warning("Sentiment column contains missing values. Dropping these rows for training.")
                    df_train = df.dropna(subset=[sentiment_column])
                else:
                    df_train = df
            else:
                df_train = df  # Use default dataset if no labels

            model, vectorizer, accuracy, precision, recall, f1 = train_sentiment_model(df_train, text_column, sentiment_column, _mapping_func=mapping_func)
            if model is None:
                return
            st.success(f"Model trained successfully!")

            # Display model metrics
            st.write("### Model Performance on Test Set:")
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**Precision (weighted):** {precision:.2f}")
            st.write(f"**Recall (weighted):** {recall:.2f}")
            st.write(f"**F1-Score (weighted):** {f1:.2f}")
            st.write(f"**Classes:** {list(model.classes_)}")

            # Predict on full dataset
            predictions, probs = predict_sentiment(df['cleaned_text'], model, vectorizer)
            df['predicted_sentiment'] = predictions
            st.write("### Predictions on Uploaded Data:")
            st.write(df[[text_column, 'predicted_sentiment']].head())

            # Sentiment counts after prediction
            display_sentiment_counts(df, 'predicted_sentiment', lambda x: x, "Sentiment Counts After Prediction")
            plot_sentiment_distribution(df, 'predicted_sentiment', lambda x: x, "Sentiment Distribution After Prediction")

            # Full dataset metrics if labeled
            if sentiment_column:
                y_true = df['sentiment_mapped']
                y_pred = df['predicted_sentiment']
                full_accuracy = accuracy_score(y_true, y_pred)
                full_precision, full_recall, full_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                st.write("### Model Performance on Full Dataset:")
                st.write(f"**Accuracy:** {full_accuracy:.2f}")
                st.write(f"**Precision (weighted):** {full_precision:.2f}")
                st.write(f"**Recall (weighted):** {full_recall:.2f}")
                st.write(f"**F1-Score (weighted):** {full_f1:.2f}")
                st.text(classification_report(y_true, y_pred))
                plot_confusion_matrix(y_true, y_pred, model.classes_)

        # Load existing model if no training
        elif model:
            predictions, probs = predict_sentiment(df['cleaned_text'], model, vectorizer)
            df['predicted_sentiment'] = predictions
            st.write("### Predictions Using Pre-Trained Model:")
            st.write(df[[text_column, 'predicted_sentiment']].head())
            display_sentiment_counts(df, 'predicted_sentiment', lambda x: x, "Sentiment Counts After Prediction")
            plot_sentiment_distribution(df, 'predicted_sentiment', lambda x: x, "Sentiment Distribution After Prediction")

        # Step 5: Sentiment Prediction on New Text
        st.header("5. Predict Sentiment on New Text 🌟")
        user_input = st.text_area("Enter text (e.g., stock news, tweet):", "sells are down, we made loss")
        if st.button("Predict Sentiment") and user_input and model:
            pred, prob = predict_sentiment([user_input], model, vectorizer)
            sentiment = pred[0]
            prob_dict = dict(zip(model.classes_, prob[0]))
            st.write(f"**Predicted Sentiment:** {sentiment.capitalize()}")
            st.write(f"**Probabilities:** {prob_dict}")
            if max(prob[0]) < 0.5:
                st.warning("Prediction confidence is low. Results may be uncertain.")

        # Download Results
        st.header("6. Download Processed Data 📥")
        output = io.BytesIO()
        df_to_download = df[[text_column, 'cleaned_text', 'predicted_sentiment']] if 'predicted_sentiment' in df.columns else df[[text_column, 'cleaned_text']]
        df_to_download.to_csv(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download Processed Data",
            data=output,
            file_name="processed_sentiment_data.csv",
            mime="text/csv"
        )

        # Documentation
        st.header("How It Works 📘")
        st.markdown("""
        - **Data Handling**: Upload datasets (CSV, Excel, TXT, JSON) with text and optional sentiment (e.g., stock news, tweets).
        - **Cleaning**: Text is normalized, tokenized, stopwords removed, and lemmatized.
        - **Model**: Logistic Regression with TF-IDF features, trained locally. Uses Sentiment140 if no labels provided.
        - **Prediction**: Analyzes new text with a 0.5 confidence threshold for clear sentiment.
        - **Results**: View counts, distributions, metrics (accuracy, precision, recall, F1), and word clouds.
        """)

if __name__ == "__main__":
    main()
