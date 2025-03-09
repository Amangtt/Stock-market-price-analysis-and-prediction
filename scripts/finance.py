import pandas as pd
import numpy as np
import plotly.express as px
import talib
import matplotlib.pyplot as plt
import pynance as pn
from scipy.stats import zscore
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

class financial_analysis:
    def __init__(self,logger):
        self.logger=logger
        self.df=None
        self.cleaned=None
    # Loading data
    def load_data(self,ticker,start_date,end_date,path):
        try:
            df=yf.download(tickers=ticker,start=start_date,end=end_date)
            df.reset_index(inplace=True)
            df.columns = df.columns.droplevel(1)
            self.df=df
            df.to_csv(path)
            self.logger.info("data loaded successfully.")
            return df
        except Exception as e:
            error_message = f"Failed to load data {e}"
            self.logger.error(error_message)

    # Checking existance of the specified columns
    def check_existance(self):
        if all(col in self.df.columns for col in ['Open', 'Close','High', 'Low','Volume' ]):
            self.logger.info("Existance Confirmed.")
            return True
        else:
            return False
        
    # Calculating techincal indicators    
    def calculate_technical_indicators(self):
        try:  
            df=self.df
            df['SMA'] = talib.SMA(df['Close'], timeperiod=10)  
            df['EMA'] = talib.EMA(df['Close'], timeperiod=10)  
            df['RSI'] = talib.RSI(df['Close'], timeperiod=10)
            macd,macd_signal,_= talib.MACD(df['Close'])
            df['MACD']=macd
            df['MACD_SIGNAL']= macd_signal
            self.cleaned = df.dropna(subset=['Close', 'SMA','RSI','MACD','MACD_SIGNAL']).reset_index()
            self.logger.info("Calculation of technical indicators successful.")
            return self.cleaned 
        except Exception as e:
            error_message = f"Failed to calculate technical indicators: {e}"
            self.logger.error(error_message)
    def closing(self,ticker): 
        df=self.df
        self.logger.info("plotting closing price.")
        df.set_index('Date', inplace=True)
        plt.figure(figsize=(10, 6))
        #Ploting closing price against date
        plt.plot(df.index,df['Close'], linestyle='-', color='b', label='Closing Price')
        plt.title(f'Closing Price Over Time for {ticker}', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.grid(True)
        plt.legend()
        # Show the plot
        plt.tight_layout()
        plt.show()

    def rolling_avg(self):
        try:  
            df=self.df
            #df.set_index('Date',inplace=True)
            window_size=7
            # Rolling avg and standard deviation after each 7 days
            df['Rolling mean']=df['Close'].rolling(window=window_size).mean()
            df['Rolling std']=df['Close'].rolling(window=window_size).std()
            plt.figure(figsize=(12,8))
            plt.plot(df.index,df['Close'],label="Closing price",c='b')
            plt.plot(df.index,df['Rolling mean'],label=f'{window_size}-Day Rolling Mean',c='r')
            plt.plot(df.index,df['Rolling std'],label=f'{window_size}-Day Rolling std',c='g')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            plt.show()
            self.logger.info("Successfully calculated and ploted rolling average.")

        except Exception as e:
            error_message = f"Failed to calculate and plot rolling average: {e}"
            self.logger.error(error_message)
    def outliers(self):
        df=self.df
        # Calculate daily returns (percentage change)
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df.dropna(subset=['Daily_Return'], inplace=True)
        # Calculate Z-scores for daily returns
        df['Z_Score'] = zscore(df['Daily_Return'])
        
        # Define a threshold for outliers (e.g., Z-score > 3 or < -3)
        threshold = 3
        df['Outlier'] = np.abs(df['Z_Score']) > threshold

        # Identify days with unusually high or low returns
        outliers = df[df['Outlier']]

        # Plot daily returns and highlight outliers
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Daily_Return'], label='Daily Returns', color='b', alpha=0.7)
        plt.scatter(outliers.index, outliers['Daily_Return'], color='r', label='Outliers', zorder=5)
        plt.axhline(threshold, color='g', linestyle='--', label=f'Threshold (Z={threshold})')
        plt.axhline(-threshold, color='g', linestyle='--')

        plt.title('Daily Returns with Outliers Highlighted', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Daily Return (%)', fontsize=14)
        plt.grid(True)
        plt.legend()
        self.logger.info("Successfully ploted outliers.")
        #ploting SMA,EMA and the others by accepting there name in the graph parameter 
    def plot_TI(self,graph):   

        plt.figure(figsize=(10, 5))
        plt.plot(self.df.index, self.df['Close'], label='Close', color='blue')
        plt.plot(self.df.index, self.df[graph], label=graph, color='orange')
        plt.show()
           
    def decompose(self):
        #df.set_index('Date',inplace=True)
        
        decomposition=seasonal_decompose(self.df['Close'],model='additive',period=30)
        # After seasonal decomposition displays trend,seasonality and resid(Irregular components that the model couldn’t explain)
        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(self.df.index,self.df["Close"], label="Original Time Series")
        plt.legend()

        plt.subplot(412)
        plt.plot(self.df.index,decomposition.trend, label="Trend", color="orange")
        plt.legend()

        plt.subplot(413)
        plt.plot(self.df.index,decomposition.seasonal, label="Seasonality", color="green")
        plt.legend()

        plt.subplot(414)
        plt.plot(self.df.index,decomposition.resid, label="Residuals", color="red")
        plt.legend()

        plt.tight_layout()
        plt.show()
        self.logger.info("Successfully Calculated seasonal decompostion and ploted Them.")
        # Financial Metrics
    def metrics(self):
        try:
            self.cleaned['Date'] = pd.to_datetime(self.cleaned['Date'])
            self.cleaned['year'] = self.cleaned['Date'].dt.year

            self.cleaned['Daily Return'] = self.cleaned['Close'].pct_change()
            self.cleaned['Cumulative Return'] = (1 + self.cleaned['Daily Return']).cumprod() - 1
            self.cleaned['Volatility'] = self.cleaned['Daily Return'].std() * np.sqrt(252) 
            plt.figure(figsize=(15, 10))

                # Plot Daily Return
            plt.subplot(3, 1, 1)
            plt.plot(self.cleaned['Date'], self.cleaned['Daily Return'], label='Daily Return', color='blue')
            plt.title('Daily Return')
            plt.ylabel('Return')
            plt.axhline(0, color='gray', linestyle='--')
            plt.legend()
            plt.grid()

            # Plot Cumulative Return
            plt.subplot(3, 1, 2)
            plt.plot(self.cleaned['Date'], self.cleaned['Cumulative Return'], label='Cumulative Return', color='green')
            plt.title('Cumulative Return')
            plt.ylabel('Cumulative Return')
            plt.axhline(0, color='gray', linestyle='--')
            plt.legend()
            plt.grid()

            # Display Volatility
            plt.subplot(3, 1, 3)
            plt.text(0.5, 0.5, f'Annualized Volatility: {self.cleaned["Volatility"].iloc[-1]:.4f}', fontsize=15, ha='center')
            plt.axis('off')
            plt.title('Volatility')

            plt.tight_layout()
            plt.show()
            self.logger.info('Successfully calculated and plotted metrics')
        except Exception as e:
            error_message = f"Failed to calculate and plot metrics: {e}"
            self.logger.error(error_message)
    
