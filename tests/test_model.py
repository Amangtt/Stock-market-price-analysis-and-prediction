import pytest
import pandas as pd
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model import Train  # Assuming the class is in train.py
from scripts.logger import SetupLogger

# Setup logger for testing
logger = SetupLogger(log_file='./logs/test.log').get_logger()

def generate_dummy_data():
    """Generate dummy stock price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    prices = np.cumsum(np.random.randn(200)) + 100  # Simulated stock prices
    return pd.DataFrame({'Date': dates, 'Close': prices})

@pytest.fixture
def train_instance():
    """Fixture to create a Train instance with dummy data."""
    data = generate_dummy_data()
    return Train(data, logger)

def test_create_sequences(train_instance):
    """Test sequence creation function."""
    X, y = train_instance.create_sequences(train_instance.df['Close'])
    assert X.shape[0] == y.shape[0]  # Ensure matching sequence lengths
    assert X.shape[1] == 60  # Check sequence window size

def test_lstm_training(train_instance):
    """Test LSTM model training."""
    train_instance.ltsm()
    assert 'model' in train_instance.model  # Ensure model is stored
    assert 'history' in train_instance.model

def test_evaluate(train_instance):
    """Test model evaluation."""
    train_instance.ltsm()
    train_instance.evaluate()
    assert 'LSTM' in train_instance.prediction  # Ensure predictions exist
    assert len(train_instance.prediction['LSTM']) > 0

def test_forecast(train_instance, tmp_path):
    """Test forecast function and CSV output."""
    output_file = tmp_path / "forecast_results.csv"
    train_instance.ltsm()
    train_instance.forecast(months=6, output_file=str(output_file))
    assert output_file.exists()  # Ensure file was created
    df = pd.read_csv(output_file)
    assert not df.empty  # Ensure file is not empty