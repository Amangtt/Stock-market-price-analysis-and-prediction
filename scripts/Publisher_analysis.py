import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
sid=SentimentIntensityAnalyzer()



class publisher:
    def __init__(self,df,logger):
        self.df=df
        self.logger=logger
    def count_publisher(self):
        """ Count the occurrences of each publisher"""
        try:
            
            count = self.df.groupby('publisher').size().reset_index(name='publisher count')
            count = count.sort_values(by='publisher count', ascending=False)
            
            # Get the top 3 publishers
            top_3 = count.head(3)

            # Filter the original DataFrame for the top 3 publishers
            top_publishers = top_3['publisher'].tolist()
            #filtered_df = df[df['publisher'].isin(top_publishers)]
            self.logger.info("Number of article publisher by a publisher calculated successfully.")
            return top_publishers
        except Exception as e:
            error_message = f"Failed to calculate number of publisher {e}"
            self.logger.error(error_message)

    def count_unique_domains(self):
        """ Count the occurrences of Unique domains to get the company with the most publish"""
        domain_count = {}

        for publisher in self.df["publisher"]:
            if "@" in publisher:  # Checking if its an email
                domain = publisher.split("@")[-1]
                domain_count[domain] = domain_count.get(domain, 0) + 1

        self.logger.info("Number of unique domains calculated successfully.")
        return domain_count
    
    def pub_days_time(self):
        """ Count the number of publish for each day to see when are most articles published"""
        try:
            df=self.df
            
            df['day_of_week'] = df['date'].dt.day_name()

            # Count the number of articles by day of the week
            day_counts = df['day_of_week'].value_counts()

            # Reorder the days for proper plotting
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = day_counts.reindex(day_order)
            self.logger.info("Ploting articles published on each day of the week")
            # Plot the results
            plt.figure(figsize=(10, 6))
            day_counts.plot(kind='bar', color='red')
            plt.title('Number of Articles Published by Day of the Week')
            plt.xlabel('Day of the Week')
            plt.ylabel('Number of Articles') 
            plt.show()
        except Exception as e:
            error_message = f"Failed to calculate number of articles published on each day of the week {e}"
            self.logger.error(error_message)

    def plot_article_by_hour(self):
        """ Count the number of publish for each hour to see when are most articles published"""
        try:
            df=self.df
            
            df['hour'] = df['date'].dt.hour
            hourly_counts = df['hour'].value_counts().sort_index()
            self.logger.info("Ploting articles published on each hour")
            # Plotting the results
            plt.figure(figsize=(10, 5))
            hourly_counts.plot(kind='bar', color='skyblue')
            plt.xlabel('Hour')
            plt.ylabel('Number of Articles')
            plt.grid(axis='y')
            plt.show()
        except Exception as e:
            error_message = f"Failed to calculate number of articles published on each hour {e}"
            self.logger.error(error_message)


    def event(self):
        """ To see which major event had alot of publish """

        try:
            df=self.df
            # Example market events
            daily_frequency = df.resample('D', on='date').size()
            market_events = {
            '2011-08-05': "U.S. Credit Rating Downgraded by S&P",
            '2012-09-13': "Federal Reserve Announces QE3",
            '2016-06-23': "Brexit Referendum",
            '2016-11-08': "U.S. Presidential Election (Donald Trump Elected)",
            '2018-03-22': "U.S. Imposes Tariffs on Chinese Goods (Start of Trade War)",
            '2020-03-23': "COVID-19 Pandemic Leads to Global Market Crash",
            '2020-04-20': "Oil Prices Turn Negative for the First Time in History",
            '2020-03-15': "Federal Reserve Slashes Interest Rates to Near Zero"
            }
            self.logger.info("Major market events against number of publication")
        
            # Plot daily publication frequency with market events
            ax = daily_frequency.plot(title='Daily Publication Frequency with Market Events', figsize=(10, 6))
            for event_date, event_name in market_events.items():
                ax.axvline(pd.to_datetime(event_date), color='skyblue', linestyle='--', label=event_name)
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.legend()
            plt.show()
        except Exception as e:
            error_message = f"Failed to calculate number major market events: {e}"
            self.logger.error(error_message)            
