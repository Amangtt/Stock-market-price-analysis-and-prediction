import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
import spacy

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class sentiment:
    def __init__(self, data, logger=None):
        self.df = data
        self.logger = logger

    def preprocess(self, text):
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            # Remove punctuation and stopwords
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            # Lemmatize
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

            if self.logger:
                self.logger.info("Word tokenization completed successfully.")

            return " ".join(tokens)  # Convert tokens back to a string
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to tokenize words: {e}")
            return ""  # Prevent returning None

    def sentiment_analysis(self, path):
        try:
            df = self.df.copy()  # Avoid modifying the original dataframe
            df['processed_headline'] = df['headline'].apply(self.preprocess)

            # Drop empty processed headlines
            df = df[df['processed_headline'] != ""]

            # Select relevant columns
            head = df[['headline', 'processed_headline', 'date', 'stock']].copy()

            # Sentiment Analysis using TextBlob
            head['score'] = head['processed_headline'].apply(lambda x: TextBlob(x).sentiment.polarity if x else 0)

            # Load spaCy model for Topic Modeling
            nlp = spacy.load("en_core_web_sm")
            head['keywords'] = head['processed_headline'].apply(lambda x: [token.text for token in nlp(x) if token.pos_ in ["NOUN", "PROPN"]] if x else [])

            # Assign sentiment labels
            head['sentiment'] = head['score'].apply(lambda score: 'Positive' if score >= 0.05 
                                                    else ('Negative' if score <= -0.05 else 'Neutral'))

            if self.logger:
                self.logger.info("Sentiment analysis completed successfully.")

            # Save to CSV
            head.to_csv(path, index=False)
            return head

        except Exception as e:
            if self.logger:
                self.logger.error(f"Sentiment analysis failed: {e}")
            return pd.DataFrame()  # Return empty DataFrame if error occurs
