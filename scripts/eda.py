import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class Descriptive:
    def __init__(self, data):
        self.df=data
      

    def headline_length(self):
  
            self.df["headline length"] = self.df["headline"].apply(len)

            self.logger.info("headline length calculated successfully.")
            return self.df
   
    def check_missing(self):

        return self.df.isnull().sum()


 
    def stock(self):

            df=self.df
            df['date'] = df['date'].str[:10]
            df['date']= pd.to_datetime(df['date'])
            stocks=df.groupby('stock').size().reset_index(name='stock count')
            stocks=stocks.sort_values(by='stock count', ascending=False)
            top_5=stocks.head(5)
            self.logger.info("Number of stocks with the most number of articles calculated successfully.")
            return top_5


    def plot_article_over_time(self):
   
            df=self.df
            """df['date'] = df['date'].str[:10]
            df['date'] = pd.to_datetime(df['date'])"""
            count_by_date= df.groupby('date').size().reset_index(name='Trend by date')
            self.logger.info("Ploting articles published overtime.")
            plt.figure(figsize=(12, 6))
            plt.plot(count_by_date['date'], count_by_date['Trend by date'], marker='o', linestyle='-', color='skyblue')
            plt.title('Number of Articles Published Over Time')
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()
  