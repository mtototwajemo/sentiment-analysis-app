**Sentiment Analysis Dashboard**

**Overview**

This repository contains a Streamlit-based sentiment analysis dashboard that allows users to analyze text data, detect sentiment labels, and visualize various insights. The application supports multiple file formats (CSV, Excel, TXT, JSON) and automatically detects text and sentiment columns in the uploaded dataset.

**Features**

Text Preprocessing: Cleans and tokenizes text, removing punctuation and stopwords.

Sentiment Mapping: Dynamically detects and maps sentiment values based on numeric ratings or categorical labels.

Automatic Column Detection: Identifies relevant text and sentiment columns from uploaded datasets.

Sentiment Model Training: Trains a Logistic Regression model using TF-IDF vectorization.

**Visualization Tools:**

Sentiment distribution plots

Word clouds for different sentiments

Text length distribution

Confusion matrix for model evaluation

Performance Metrics: Provides accuracy, precision, recall, and F1-score for sentiment classification.

Model Persistence: Saves trained models using joblib for future predictions.

Pre-Trained Model Support: Users can load and apply pre-trained models for sentiment analysis.

**Installation**

To run this application, ensure you have Python installed and set up a virtual environment.

# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

**Running the Application
**
To start the Streamlit dashboard, use the following command:

streamlit run app.py

**Usage**

Upload a dataset containing text and sentiment labels.

The app will automatically detect the relevant columns.

Preprocess and clean text data.

Train a sentiment analysis model or load an existing one.

Generate visualizations and analyze sentiment trends.

Export results if needed.

**Dependencies**

Python 3.7+

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

NLTK

Scikit-learn

WordCloud

Joblib

JSON

Logging

**Contributing**

If you want to improve this project, feel free to fork the repository and submit a pull request.

**License**

This project is licensed under the MIT License.

**Contact
**
For any queries, reach out via GitHub Issues or email.
