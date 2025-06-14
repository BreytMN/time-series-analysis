import numpy as np
import pandas as pd  # type: ignore
import pytest

from time_series_analysis import TimeSeriesAnalyzer

DAYS_IN_YEAR = 365


@pytest.fixture
def tsa() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=DAYS_IN_YEAR)
    values = np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    data = {"unique_id": "id", "ds": dates, "y": values}
    return TimeSeriesAnalyzer(pd.DataFrame(data))


def test_tsa_properties(tsa: TimeSeriesAnalyzer) -> None:
    tsa._frequency_analysis_cache["some"] = "thing"  # type: ignore
    assert tsa._oot_periods == 0
    assert tsa._frequency_analysis_cache
    tsa.oot_periods = 1
    assert tsa._oot_periods == 1
    assert not tsa._frequency_analysis_cache


def test_tsa_data(tsa: TimeSeriesAnalyzer) -> None:
    tsa.oot_periods = 0
    assert len(tsa.dates) == DAYS_IN_YEAR
    assert len(tsa.series) == DAYS_IN_YEAR
    assert len(tsa.train_data) == DAYS_IN_YEAR
    assert len(tsa.oot_data) == 0

    tsa.oot_periods = 7
    assert len(tsa.dates) == (DAYS_IN_YEAR - 7)
    assert len(tsa.series) == (DAYS_IN_YEAR - 7)
    assert len(tsa.train_data) == (DAYS_IN_YEAR - 7)
    assert len(tsa.oot_data) == 7


def test_tsa_plot_trend_raises(tsa: TimeSeriesAnalyzer) -> None:
    msg = "At least one of polyfit_degrees or moving_average_windows must be provided"
    with pytest.raises(ValueError, match=msg):
        tsa.plot_trends()


def test_tsa__repr_mimebundle_(tsa: TimeSeriesAnalyzer) -> None:
    tsa.oot_periods = 1
    assert isinstance(tsa._repr_mimebundle_(), dict)
