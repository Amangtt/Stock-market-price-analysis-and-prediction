import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.eda import Descriptive  

@pytest.fixture
def sample_data():
    data = {
        "headline": ["Tesla Stock Surges", "Market Crash Expected", "AI Revolution Continues", "Oil Prices Drop", "Tech Boom Ahead"],
        "url": ["https://example.com/tesla","https://example.com/market-crash","https://example.com/ai-revolution","https://example.com/oil-prices","https://example.com/tech-boom"],
        "publisher": ["Reuters", "CNBC", "Bloomberg", "CNBC", "Reuters"],
        "stock": ["TSLA", "SPY", "NVDA", "OIL", "AAPL"],
        "date": ["2025-03-01 10:00:00", "2025-03-02 11:30:00", "2025-03-03 09:45:00", "2025-03-01 15:20:00", "2025-03-02 08:10:00"]
    }
    df = pd.DataFrame(data)
    return df

def test_headline_length(sample_data):
    desc = Descriptive(sample_data)
    df_result = desc.headline_length()
    assert "headline length" in df_result.columns, "Column 'headline length' should exist"
    assert df_result["headline length"].iloc[0] == len(sample_data["headline"][0]), "Incorrect headline length calculated"

def test_check_missing(sample_data):
    sample_data.loc[0, "headline"] = None  # Introduce a missing value
    desc = Descriptive(sample_data)
    missing_values = desc.check_missing()
    assert missing_values["headline"] == 1, "Missing values count should be correctly identified"


def test_stock(sample_data):
    desc = Descriptive(sample_data)
    top_stocks = desc.stock()
    assert len(top_stocks) <= 5, "Should return at most 5 stock entries"
    assert "stock" in top_stocks.columns and "stock count" in top_stocks.columns, "Missing expected columns"

def test_plot_article_over_time(sample_data):
    desc = Descriptive(sample_data)
    try:
        desc.plot_article_over_time()  # Ensure no errors occur during execution
    except Exception as e:
        pytest.fail(f"plot_article_over_time() raised an error: {e}")
