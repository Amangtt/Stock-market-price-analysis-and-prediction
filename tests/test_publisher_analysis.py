import pandas as pd
import pytest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.Publisher_analysis import publisher  

@pytest.fixture
def sample_data():
    """Create a sample DataFrame with proper columns for testing."""
    data = {
        'headline': [
            'Stocks That Hit 52-Week Highs On Friday', 'Tech Stocks Rally',
            'Market Update: Oil Prices Surge', 'Fed Announces Rate Cut'
        ],
        'url': [
            'https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday',
            'https://example.com/tech-stocks-rally',
            'https://example.com/oil-prices-surge',
            'https://example.com/fed-rate-cut'
        ],
        'publisher': [
            'Benzinga Insights', 'CNBC', 'Reuters', 'CNBC'
        ],
        'date': pd.to_datetime([
            '2020-06-05 10:30:54-04:00', '2020-06-05 14:00:00-04:00',
            '2020-06-06 09:15:00-04:00', '2020-06-07 16:45:00-04:00'
        ]),
        'stock': ['AAPL', 'GOOGL', 'OIL', 'FED']
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_logger():
    """Create a mock logger to prevent real logging during tests."""
    return MagicMock()

@pytest.fixture
def publisher_instance(sample_data, mock_logger):
    """Create an instance of the publisher class with sample data and mock logger."""
    return publisher(sample_data, mock_logger)

def test_count_publisher(publisher_instance):
    """Test counting the top publishers."""
    result = publisher_instance.count_publisher()
    assert isinstance(result, list)
    assert len(result) == 3
    

def test_count_unique_domains(publisher_instance):
    """Test counting unique domains from publisher email addresses."""
    result = publisher_instance.count_unique_domains()
    assert isinstance(result, dict)
    assert len(result) == 0  # No email-based publishers in test data

def test_pub_days_time(publisher_instance):
    """Ensure pub_days_time runs without exceptions."""
    try:
        publisher_instance.pub_days_time()
    except Exception as e:
        pytest.fail(f"pub_days_time() raised an exception: {e}")

def test_plot_article_by_hour(publisher_instance):
    """Ensure plot_article_by_hour runs without errors."""
    try:
        publisher_instance.plot_article_by_hour()
    except Exception as e:
        pytest.fail(f"plot_article_by_hour() raised an exception: {e}")

def test_event(publisher_instance):
    """Ensure the event method runs successfully."""
    try:
        publisher_instance.event()
    except Exception as e:
        pytest.fail(f"event() raised an exception: {e}")
