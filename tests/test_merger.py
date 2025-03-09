import pytest
import pandas as pd
import numpy as np
import logging
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.task3_merger import merge  

# Configure a simple logger for testing
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@pytest.fixture
def sample_data():
    """Fixture to create sample news and stock data."""
    news_data = pd.DataFrame({
        'date': ['2025-03-01', '2025-03-02', '2025-03-02', '2025-03-03'],
        'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive']
    })

    stock_data = pd.DataFrame({
        'Date': ['2025-03-01', '2025-03-02', '2025-03-03', '2025-03-04'],
        'Close': [100, 102, 101, 105]
    })

    return news_data, stock_data

@pytest.fixture
def merger_instance(sample_data):
    """Fixture to create a merge instance with sample data."""
    news, stock = sample_data
    return merge(news, stock, logger)

def test_combine(merger_instance):
    """Test the combine() function merges news and stock correctly."""
    merged = merger_instance.combine()
    assert merged is not None
    assert 'sentiment_score' in merged.columns
    assert merged.shape[0] > 0  # Ensure data was merged
    print(merged.head())

def test_final(merger_instance):
    """Test the final() function calculates returns and sentiment shift."""
    merger_instance.combine()  # Ensure merge happens first
    result = merger_instance.final()
    
    assert result is not None
    assert 'Daily Return' in result.columns
    assert 'Lagged Sentiment' in result.columns
    assert result.shape[0] > 0  # Ensure there's valid data
    print(result.head())

def test_compute_correlation(merger_instance):
    """Test compute_correlation() returns a valid value."""
    merger_instance.combine()
    merger_instance.final()
    correlation = merger_instance.compute_correlation()

    if correlation is None or np.isnan(correlation):
        pytest.skip("Not enough data for correlation.")
    else:
        assert -1 <= correlation <= 1, "Correlation is out of bounds"
    print(f"Computed Correlation: {correlation}")
