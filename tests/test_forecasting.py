import pytest
import pandas as pd
import numpy as np
import os,sys
from unittest.mock import MagicMock
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.future_forecast import ModelForecaster  # Update with the correct import path

@pytest.fixture
def mock_logger():
    """Mock logger to avoid real logging in tests."""
    return MagicMock()

@pytest.fixture
def sample_csv_files(tmp_path):
    """Create sample CSV files for testing."""
    # Create historical data CSV
    historical_data = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
        "Close": np.linspace(100, 110, 10)
    })
    historical_csv = tmp_path / "historical.csv"
    historical_data.to_csv(historical_csv, index=False)

    # Create forecast data CSV
    forecast_data = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-11", periods=5, freq="D"),
        "forecast": np.linspace(110, 115, 5),
        "conf_lower": np.linspace(108, 113, 5),
        "conf_upper": np.linspace(112, 117, 5)
    })
    forecast_csv = tmp_path / "forecast.csv"
    forecast_data.to_csv(forecast_csv, index=False)

    return str(historical_csv), str(forecast_csv)

def test_model_forecaster_initialization(mock_logger, sample_csv_files):
    """Test if ModelForecaster initializes correctly."""
    historical_csv, forecast_csv = sample_csv_files
    model = ModelForecaster(mock_logger, historical_csv, forecast_csv)
    
    assert model.historical_data is not None
    assert model.forecast_data is not None
    assert "forecast" in model.forecast_data.columns
    mock_logger.info.assert_called_with("Data loaded successfully")

def test_analyze_forecast(mock_logger, sample_csv_files):
    """Test the analyze_forecast method for expected output."""
    historical_csv, forecast_csv = sample_csv_files
    model = ModelForecaster(mock_logger, historical_csv, forecast_csv)

    analysis_df = model.analyze_forecast()
    
    assert isinstance(analysis_df, pd.DataFrame)
    assert "Trend" in analysis_df.columns
    assert "Volatility_Level" in analysis_df.columns
    mock_logger.info.assert_any_call("successfully interpreted forecast results")

def test_plot_forecast(mock_logger, sample_csv_files):
    """Test the plot_forecast method runs without error."""
    historical_csv, forecast_csv = sample_csv_files
    model = ModelForecaster(mock_logger, historical_csv, forecast_csv)

    try:
        model.plot_forecast()
        mock_logger.info.assert_called_with("successfully plotted forecast")
    except Exception as e:
        pytest.fail(f"plot_forecast() raised an exception: {e}")
