from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
import spacy
#nltk.download('vader_lexicon')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
# nltk.download('punkt_tab')
# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class sentiment:
        def __init__(self,data,logger=None):
                self.df=data
                self.logger=logger
        def preprocess(self,text):
                try:
                        # Tokenize
                        tokens = word_tokenize(text.lower())
                        # Remove punctuation and stopwords
                        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
                        # Lemmatize
                        tokens = [lemmatizer.lemmatize(word) for word in tokens]
                        self.logger.info("Word tokenization completed successfully.")
                        return " ".join(tokens)  # Convert tokens back to a string
                
                except Exception as e:
                        error_message = f"Failed to tokenize words: {e}"
                        self.logger.error(error_message)

        def sentiment_analysis(self,path):
        # Apply preprocessing
                try:
                        df=self.df
                        df['processed_headline'] = df['headline'].apply(self.preprocess)

                        # Select first 50 rows as a DataFrame
                        head = df[['headline', 'processed_headline','date','stock']].copy()

                        # Sentiment Analysis using TextBlob
                        head['score'] = head['processed_headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

                        # Load spaCy model for Topic Modeling
                        nlp = spacy.load("en_core_web_sm")
                        head['keywords'] = head['processed_headline'].apply(lambda x: [token.text for token in nlp(x) if token.pos_ in ["NOUN", "PROPN"]])

                        # Assign sentiment labels
                        head['sentiment'] = head['score'].apply(lambda score: 'Positive' if score >= 0.05 
                                                                else ('Negative' if score <= -0.05 else 'Neutral'))
                        self.logger.info("Sentiment analysis completed successfully.")
                        # Print output
                        head.to_csv(path)
                        return head
                except Exception as e:
                        error_message = f"Sentiment analysis falied: {e}"
                        self.logger.error(error_message)        
                



