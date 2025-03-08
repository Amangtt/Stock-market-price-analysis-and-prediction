import pandas as pd
import pytest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.Sentiment_analysis import sentiment  

@pytest.fixture
def sample_data():
    """Create a sample DataFrame with realistic text headlines."""
    data = {
        'headline': [
            "Stocks soar as tech leads market rally!",
            "Oil prices crash due to global oversupply.",
            "Fed announces new policies to stabilize economy.",
            "Investors cautious amid economic uncertainty."
        ],
        'date': pd.to_datetime([
            '2023-05-10 14:30:00', '2023-05-11 09:45:00',
            '2023-05-12 11:00:00', '2023-05-13 16:20:00'
        ]),
        'stock': ['AAPL', 'OIL', 'FED', 'SPY']
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_logger():
    """Create a mock logger to prevent real logging during tests."""
    return MagicMock()

@pytest.fixture
def sentiment_instance(sample_data, mock_logger):
    """Create an instance of the sentiment class with sample data and mock logger."""
    return sentiment(sample_data, mock_logger)

def test_preprocess(sentiment_instance):
    """Test text preprocessing function."""
    text = "Stocks are rallying!!! This is amazing, but investors remain cautious."
    processed_text = sentiment_instance.preprocess(text)
    
    assert isinstance(processed_text, str)
    assert "stock" in processed_text  # Ensuring key words are retained
    assert "rallying" in processed_text  # Lemmatized form should remain
    assert "amazing" in processed_text  # Stopwords should be removed

def test_sentiment_analysis(sentiment_instance, tmp_path):
    """Test sentiment analysis function."""
    test_output_path = tmp_path / "sentiment_output.csv"
    
    result_df = sentiment_instance.sentiment_analysis(test_output_path)
    
    assert isinstance(result_df, pd.DataFrame)
    assert 'score' in result_df.columns
    assert 'sentiment' in result_df.columns
    assert 'keywords' in result_df.columns
    assert test_output_path.exists()  # Ensure the file was created

    # Check that sentiment labels are assigned
    assert set(result_df['sentiment']).issubset({'Positive', 'Negative', 'Neutral'})
