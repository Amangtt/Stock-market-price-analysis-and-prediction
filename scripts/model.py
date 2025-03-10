import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class Train:
    def __init__(self,data,logger=None,column='Close'):
        self.df = data
        self.column = column
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logger=logger
        self.scaler=StandardScaler()
        self.model = {}
        self.prediction = {}
    def create_sequences(self,data, window=60):
            """ Creating sequence for LTSM"""
            try:
                data = self.scaler.fit_transform(data.values.reshape(-1,1))
                X, y = [], []
                for i in range(len(data)-window):
                    X.append(data[i:i+window])
                    y.append(data[i+window])
                self.logger.info("successfully created sequence.")
                return np.array(X), np.array(y)
            except Exception as e:
                error_message = f"Failed to create sequence{e}"
                self.logger.error(error_message)

    def ltsm(self):
            """LTSM model """
            try:
                data = self.df[self.column]
                #data.set_index('Date', inplace=True) " # Set as index

                # Prepare data

                X, y= self.create_sequences(data)
                self.X_train, self.X_test = X[:-60], X[-60:]
                self.y_train, self.y_test = y[:-60], y[-60:]
                model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')

                # Train
                history=model.fit(self.X_train, self.y_train, epochs=50, batch_size=32,
                                validation_split=0.2, verbose=0)
                self.model = {'model': model, 'history': history}
                self.last_sequence = self.X_test[-1] 
                self.logger.info("successfully fitted ltsm model.")
            except Exception as e:
                error_message = f"Failed to train model{e}"
                self.logger.error(error_message)
    def evaluate(self):
            try:
                model = self.model['model'] 
                lstm_forecast = model.predict(self.X_test)
                y_test=self.y_test
                print(f'MAE:{mean_absolute_error(y_test,lstm_forecast)}')
                print(f'MSE:{mean_squared_error(y_test,lstm_forecast)}')
                print(f'MAPE:{np.mean(np.abs((y_test- lstm_forecast) / y_test)) * 100}')
                lstm_forecast = self.scaler.inverse_transform(lstm_forecast).flatten()
                self.prediction['LSTM']=lstm_forecast
                y_test=self.scaler.inverse_transform(self.y_test).flatten()
                self.y_test=y_test
                self.logger.info("successfully evaluated the model.")
            except Exception as e:
                error_message = f"Failed to evaluate the model{e}"
                self.logger.error(error_message)
    def plot(self):
        """Plotting the prediction of the models"""
        try:
            plt.plot(self.y_test, label='Actual')
            plt.plot(self.prediction['LSTM'], label='LSTM Forecast', linestyle='--')
            plt.title('TSLA Stock Price Forecast (LSTM)')
            plt.legend()
            plt.show()
            self.logger.info("successfully plotted the prediction.")
        except Exception as e:
            error_message = f"Failed to plot the prediction{e}"
            self.logger.error(error_message)

    def forecast(self, months=12, output_file='forecast_results.csv'):
        """
        Generate forecasts for 6 to 12 months into the future using the trained LSTM model and save to CSV.
        """
        self.df['Date'] = pd.to_datetime(self.df['Date'])  # Convert Date column to datetime
        self.df.set_index('Date', inplace=True)

        # Validate input months
        if months not in [6, 12]:
            raise ValueError("Months should be either 6 or 12.")

        # Ensure data is sorted in ascending order by date
        if self.df.index.is_monotonic_decreasing:
            self.df = self.df.sort_index(ascending=True)

        # Convert months to trading days (assuming 21 trading days per month)
        periods = 21 * months

        try:
            # Prepare forecast date range
            forecast_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
            forecast_data = pd.DataFrame(index=forecast_dates, columns=['forecast', 'conf_lower', 'conf_upper'])

            # Check if the LSTM model is available
            if 'model' not in self.model:
                raise ValueError("LSTM model is not trained or available.")

            predictions = []
            residuals = []
            current_sequence = self.last_sequence.copy()  # Use the last sequence as the starting point

            for _ in range(periods):
                # Predict the next value
                next_pred = self.model['model'].predict(current_sequence[np.newaxis, :, :])[0][0]
                predictions.append(next_pred)

                # Estimate residuals based on the last available training data
                actual_last_value = self.X_train[-1]  # Assuming train data contains past actual values
                residuals.append(actual_last_value - next_pred)

                # Update the sequence with the predicted value
                current_sequence = np.roll(current_sequence, -1)  # Shift the sequence to the left
                current_sequence[-1] = next_pred  # Replace the last value with the predicted value

            # Convert predictions to DataFrame
            forecast_data['forecast'] = predictions

            # Compute Confidence Intervals (95%)
            std_dev = np.std(residuals)
            z_score = norm.ppf(0.975)  # 1.96 for 95% CI
            forecast_data['conf_lower'] = forecast_data['forecast'] - (z_score * std_dev)
            forecast_data['conf_upper'] = forecast_data['forecast'] + (z_score * std_dev)

            # Apply inverse transform
            scaled_values = forecast_data[['forecast', 'conf_lower', 'conf_upper']].values
            scaled_values = scaled_values.reshape(-1, 3)  # Ensure 2D shape
            forecast_data[['forecast', 'conf_lower', 'conf_upper']] = self.scaler.inverse_transform(scaled_values)

            # Save forecast results to CSV
            forecast_data.to_csv(output_file)
            self.logger.info("Successfully generated and saved LSTM forecasts.")
        except Exception as e:
            error_message = f"Forecasting failed: {e}"
            self.logger.error(error_message)
            raise ValueError(error_message)



"""d=pd.read_csv('./Data/AAPL.csv')
t=Train(d,logger)
t.ltsm()
t.evaluate()
t.plot()"""