import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class merge:
    def __init__(self,news,stock,logger):
        self.news=news
        self.stock=stock
        self.logger=logger
        self.merged_data=None
        self.df=None

    def combine(self):
        try:
            # Convert dates to datetime
            self.news['date'] = self.news['date'].str[:10]
            self.news['date'] = pd.to_datetime(self.news['date'], errors='coerce')
            self.stock['Date'] = pd.to_datetime(self.stock['Date'], errors='coerce')

            # Filter stock data by ticker
            stock_filtered = self.stock  
            stock_filtered['Date'] = stock_filtered['Date'].dt.date
            self.news['date'] = self.news['date'].dt.date

            # Convert sentiment labels to numerical scores
            sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            self.news['sentiment_score'] = self.news['sentiment'].map(sentiment_mapping)

            # Compute average sentiment per day
            news_grouped = self.news.groupby('date', as_index=False)['sentiment_score'].mean()

            # Merge averaged news sentiment with stock data
            self.merged_data = pd.merge(news_grouped, stock_filtered, left_on='date', right_on='Date', how='inner')
            self.merged_data.drop(columns=['Date'], inplace=True)  # Drop duplicate date column
            self.logger.info("successfully Combined news and stock datasets.")
            return self.merged_data
        
        except Exception as e:
            error_message = f"Failed to combine news and stock {e}"
            self.logger.error(error_message)

    def final(self):
        try:
            self.df = self.merged_data.copy()

            # Convert stock prices to numeric
            self.df['Close'] = pd.to_numeric(self.df['Close'], errors='coerce')

            # Compute daily stock returns
            self.df['Daily Return'] = self.df['Close'].pct_change()

            # Lag the sentiment score by 1 day
            self.df['Lagged Sentiment'] = self.df['sentiment_score'].shift(1)

            # Drop NaN values
            self.df.dropna(subset=['Daily Return', 'Lagged Sentiment'], inplace=True)
            self.logger.info("successfully calculated daily return and lagged sentiment.")

            return self.df[['date', 'Close', 'sentiment_score', 'Daily Return', 'Lagged Sentiment']]
        except Exception as e:
            error_message = f"Failed to calculate daily return and lagged sentiment {e}"
            self.logger.error(error_message)
   
    def compute_correlation(self):
        try:
            # Ensure numerical data is used
            if self.df is None or self.df.empty:
                self.logger.warning("Dataframe is empty. Correlation cannot be computed.")
                return None

            # Drop rows with NaN values in relevant columns
            valid_data = self.df[['Lagged Sentiment', 'Daily Return']].dropna()
            
            if valid_data.shape[0] < 2:  # Correlation needs at least 2 data points
                self.logger.warning("Not enough valid data points for correlation calculation.")
                return None

            correlation = valid_data.corr().iloc[0, 1]
            print(f"Correlation between sentiment score and stock movement: {correlation:.4f}")
            self.logger.info("Successfully calculated correlation")
            return correlation

        except Exception as e:
            error_message = f"Failed to calculate correlation {e}"
            self.logger.error(error_message)
            return None