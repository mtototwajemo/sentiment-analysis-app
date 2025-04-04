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
import chardet

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Color palette
SENTIMENT_COLORS = {'positive': '#00cc96', 'negative': '#ef476f', 'neutral': '#ffd166', 'good': '#00cc96', 'bad': '#ef476f'}

# Enhanced custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
        color: #2b2d42;
    }
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.05);
        max-width: 1300px;
        margin: 2rem auto;
    }
    h1 {
        color: #1d3557;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    h2 {
        color: #457b9d;
        font-size: 1.9em;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #a8dadc;
        font-weight: 500;
    }
    .stButton>button {
        background: linear-gradient(90deg, #457b9d 0%, #1d3557 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-size: 1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d3557 0%, #457b9d 100%);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>label, .stSelectbox>label, .stCheckbox>label {
        color: #1d3557;
        font-weight: 500;
        font-size: 1.1em;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 0.5rem;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }
    .stExpander {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Utility Functions ---

def clean_text(text, remove_stopwords=True, lemmatize=True):
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
    values = df[sentiment_column].dropna().astype(str).str.lower()
    unique_values = set(values)

    try:
        numeric_values = pd.to_numeric(values, errors='coerce')
        if numeric_values.notna().all():
            min_val, max_val = numeric_values.min(), numeric_values.max()
            if min_val >= 1 and max_val <= 5:
                st.info("Detected 1-5 rating scale: 1-2 = negative, 3 = neutral, 4-5 = positive")
                return lambda x: 'negative' if pd.to_numeric(x) <= 2 else ('neutral' if pd.to_numeric(x) == 3 else 'positive')
            elif min_val >= 0 and max_val <= 1:
                st.info("Detected 0-1 score: <0.4 = negative, 0.4-0.6 = neutral, >0.6 = positive")
                return lambda x: 'negative' if pd.to_numeric(x) < 0.4 else ('neutral' if pd.to_numeric(x) <= 0.6 else 'positive')
            else:
                st.info("Detected numeric values; mapping based on terciles.")
                terciles = np.percentile(numeric_values, [33, 66])
                return lambda x: 'negative' if pd.to_numeric(x) <= terciles[0] else ('neutral' if pd.to_numeric(x) <= terciles[1] else 'positive')
    except:
        pass

    common_labels = {'positive', 'negative', 'neutral', 'pos', 'neg', 'neu', 'good', 'bad'}
    if unique_values.issubset(common_labels):
        st.info(f"Detected categorical labels: {unique_values}")
        return lambda x: 'positive' if str(x).lower() in ['positive', 'pos', 'good'] else ('negative' if str(x).lower() in ['negative', 'neg', 'bad'] else 'neutral')

    if len(unique_values) <= 5:
        st.info(f"Detected custom categorical labels: {unique_values}. Using as-is.")
        return lambda x: str(x).lower()

    st.error(f"Cannot interpret '{sentiment_column}' with values: {unique_values}. Use ratings, scores, or clear labels.")
    return None

def detect_columns(df):
    text_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['text', 'review', 'comment', 'description', 'news', 'tweet'])]
    sentiment_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sentiment', 'label', 'target', 'class', 'score', 'rating'])]
    text_column = text_candidates[0] if text_candidates else df.select_dtypes(include=['object']).columns[0]
    sentiment_column = sentiment_candidates[0] if sentiment_candidates else None
    return text_column, sentiment_column

def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding']

def load_file(uploaded_file):
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
            st.error("Unsupported file format. Use CSV, Excel, TXT, or JSON.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error loading file: {e}")
        return None

# --- Model Training and Prediction ---

@st.cache_resource
def train_sentiment_model(df_train, text_column, sentiment_column, _mapping_func=None):
    if sentiment_column is None or _mapping_func is None:
        st.warning("No labeled data found. Using Sentiment140 sample dataset.")
        try:
            default_data = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitt30k.csv', encoding='latin-1')
            default_data.columns = ['text', 'sentiment']
            default_data['sentiment'] = default_data['sentiment'].map({0: 'negative', 4: 'positive'}).fillna('neutral')
            df_train = default_data.sample(1000, random_state=42)
            text_column, sentiment_column = 'text', 'sentiment'
            _mapping_func = lambda x: x
        except:
            st.error("Failed to load default dataset. Provide labeled data to proceed.")
            return None, None, 0, 0, 0, 0

    df_train = df_train.dropna(subset=[text_column])
    df_train['cleaned_text'] = df_train[text_column].apply(lambda x: clean_text(x))
    df_train['sentiment_mapped'] = df_train[sentiment_column].apply(_mapping_func)

    st.write("Training Data Sentiment Distribution:")
    st.write(df_train['sentiment_mapped'].value_counts())

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df_train['cleaned_text'])
    y = df_train['sentiment_mapped']

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    return model, vectorizer, accuracy, precision, recall, f1

def load_model_and_vectorizer():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except FileNotFoundError:
        return None, None
    return model, vectorizer

def predict_sentiment(texts, model, vectorizer, threshold=0.5):
    cleaned_texts = [clean_text(text) for text in texts]
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

# --- Visualization Functions ---

def plot_sentiment_distribution(df, sentiment_column, mapping_func, title="Sentiment Distribution"):
    df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='sentiment_mapped', data=df, ax=ax, palette=SENTIMENT_COLORS, hue='sentiment_mapped', legend=False)
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_word_cloud(df, text_column, sentiment_column, sentiment, mapping_func):
    df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
    words = ' '.join(df[df['sentiment_mapped'] == sentiment][text_column].dropna())
    if words:
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words,
                            colormap='viridis').generate(words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{sentiment.capitalize()} Sentiment Words', fontsize=16)
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

def plot_text_length_distribution(df, text_column):
    df['text_length'] = df[text_column].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['text_length'], bins=30, kde=True, ax=ax, color='#457b9d')
    ax.set_title("Text Length Distribution", fontsize=16)
    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def display_sentiment_counts(df, sentiment_column, mapping_func, title="Sentiment Counts"):
    df['sentiment_mapped'] = df[sentiment_column].apply(mapping_func)
    counts = df['sentiment_mapped'].value_counts()
    st.write(f"**{title}:**")
    for sentiment, count in counts.items():
        st.write(f"{sentiment.capitalize()}: {count}")

# --- Main Application ---

def main():
    st.title("Sentiment Analysis Dashboard")
    st.markdown("Analyze sentiments in stock news, tweets, or reports with a sleek, intuitive interface.")

    # Sidebar with usage note
    st.sidebar.title("How It Works")
    st.sidebar.markdown("""
        This app guides you through a complete sentiment analysis workflow:
        - Upload your dataset (CSV, Excel, TXT, JSON).
        - Explore data with visualizations.
        - Preprocess text data.
        - Train a sentiment model.
        - Predict sentiments on new text.
        - Download results.
        Scroll through the sections below to get started!
    """)

    # Section 1: Data Upload
    st.header("1. Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV, Excel, TXT, or JSON", type=["csv", "xlsx", "txt", "json"])

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is None or df.empty:
            st.error("Failed to load dataset or dataset is empty.")
            return

        st.subheader("Data Preview")
        st.dataframe(df.head())

        text_column, sentiment_column = detect_columns(df)
        st.write(f"Detected: Text = {text_column}, Sentiment = {sentiment_column}")

        text_column = st.selectbox("Select Text Column:", df.columns, index=df.columns.get_loc(text_column))
        sentiment_column = st.selectbox("Select Sentiment Column (optional):", [None] + list(df.columns),
                                        index=(df.columns.get_loc(sentiment_column) + 1 if sentiment_column else 0))

        # Section 2: EDA
        st.header("2. Exploratory Data Analysis")
        with st.expander("Dataset Summary"):
            st.write(df.describe(include="all"))
            st.write("Missing Values:")
            st.write(df.isnull().sum())

        mapping_func = None
        if sentiment_column:
            mapping_func = detect_and_map_sentiment(df, sentiment_column)
            if mapping_func is None:
                return
            display_sentiment_counts(df, sentiment_column, mapping_func)
            plot_sentiment_distribution(df, sentiment_column, mapping_func)

            st.subheader("Word Clouds")
            col1, col2 = st.columns(2)
            with col1:
                plot_word_cloud(df, text_column, sentiment_column, 'positive', mapping_func)
            with col2:
                plot_word_cloud(df, text_column, sentiment_column, 'negative', mapping_func)

        plot_text_length_distribution(df, text_column)

        # Section 3: Preprocessing
        st.header("3. Data Preprocessing")
        remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        lemmatize = st.checkbox("Lemmatize Words", value=True)
        df['cleaned_text'] = df[text_column].apply(lambda x: clean_text(x, remove_stopwords, lemmatize))

        if st.checkbox("Preview Cleaned Data"):
            st.write(df[[text_column, 'cleaned_text']].head())

        # Section 4: Model Training
        st.header("4. Train Sentiment Model")
        model, vectorizer = load_model_and_vectorizer()

        if st.button("Train Model"):
            df_train = df.dropna(subset=[sentiment_column]) if sentiment_column else df
            model, vectorizer, accuracy, precision, recall, f1 = train_sentiment_model(df_train, text_column, sentiment_column, mapping_func)
            if model is None:
                return
            st.success("Model trained successfully!")
            st.write(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

            predictions, probs = predict_sentiment(df['cleaned_text'], model, vectorizer)
            df['predicted_sentiment'] = predictions
            st.write("Predictions:")
            st.write(df[[text_column, 'predicted_sentiment']].head())
            display_sentiment_counts(df, 'predicted_sentiment', lambda x: x)
            plot_sentiment_distribution(df, 'predicted_sentiment', lambda x: x)

            if sentiment_column:
                y_true = df['sentiment_mapped']
                y_pred = df['predicted_sentiment']
                st.text(classification_report(y_true, y_pred))
                plot_confusion_matrix(y_true, y_pred, model.classes_)

        elif model:
            predictions, probs = predict_sentiment(df['cleaned_text'], model, vectorizer)
            df['predicted_sentiment'] = predictions
            st.write("Predictions with Pre-trained Model:")
            st.write(df[[text_column, 'predicted_sentiment']].head())
            plot_sentiment_distribution(df, 'predicted_sentiment', lambda x: x)

        # Section 5: Predict New Text
        st.header("5. Predict Sentiment on New Text")
        user_input = st.text_area("Enter text:", "Sales are down, we made a loss")
        if st.button("Predict") and user_input and model:
            pred, prob = predict_sentiment([user_input], model, vectorizer)
            st.write(f"Sentiment: {pred[0].capitalize()}")
            st.write(f"Probabilities: {dict(zip(model.classes_, prob[0]))}")

        # Section 6: Download
        st.header("6. Download Results")
        output = io.BytesIO()
        df_to_download = df[[text_column, 'cleaned_text', 'predicted_sentiment']] if 'predicted_sentiment' in df.columns else df[[text_column, 'cleaned_text']]
        df_to_download.to_csv(output, index=False)
        output.seek(0)
        st.download_button("Download CSV", output, "processed_data.csv", "text/csv")

if __name__ == "__main__":
    main()