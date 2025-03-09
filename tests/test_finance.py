import pytest
import logging
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.finance import financial_analysis
import pandas as pd

# Create a mock logger
class MockLogger:
    def __init__(self):
        self.messages = []
    def info(self, msg):
        self.messages.append(f"INFO: {msg}")
    def error(self, msg):
        self.messages.append(f"ERROR: {msg}")

@pytest.fixture
def analysis():
    logger = MockLogger()
    return financial_analysis(logger)

def test_load_data(analysis):
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-03-01"
    path = "test_data.csv"
    
    df = analysis.load_data(ticker, start_date, end_date, path)
    
    assert isinstance(df, pd.DataFrame)
    assert "Close" in df.columns
    assert os.path.exists(path)
    
    os.remove(path)  # Clean up after test

def test_check_existance(analysis):
    analysis.df = pd.DataFrame({
        "Open": [1, 2, 3],
        "Close": [4, 5, 6],
        "High": [7, 8, 9],
        "Low": [10, 11, 12],
        "Volume": [100, 200, 300]
    })
    assert analysis.check_existance() == True


def test_outliers(analysis):
    analysis.df = pd.DataFrame({
        "Close": [100, 110, 90, 200, 85, 95, 150, 105, 115, 120]
    })
    analysis.outliers()
    
    assert "Daily_Return" in analysis.df.columns
    assert "Z_Score" in analysis.df.columns
    assert "Outlier" in analysis.df.columns

def test_metrics(analysis):
    analysis.cleaned = pd.DataFrame({
        "Date": pd.date_range(start="2024-01-01", periods=10),
        "Close": [100, 102, 101, 103, 105, 108, 110, 112, 115, 118]
    })
    analysis.metrics()
    
    assert "Daily Return" in analysis.cleaned.columns
    assert "Cumulative Return" in analysis.cleaned.columns
    assert "Volatility" in analysis.cleaned.columns
