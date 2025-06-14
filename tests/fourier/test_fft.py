import numpy as np
import pandas as pd  # type: ignore
import pytest

from time_series_analysis.fourier import FFTAnalysis
from time_series_analysis.shared.interfaces import Detrend


@pytest.fixture
def constant() -> pd.Series:
    """Fixture for a constant series (DC signal)."""
    return pd.Series([5.0] * 100)


@pytest.fixture
def sine_wave_10() -> pd.Series:
    """Fixture for a sine wave with period 10."""
    return pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))


@pytest.fixture
def sine_wave_20() -> pd.Series:
    """Fixture for a sine wave with period 20."""
    return pd.Series(np.sin(2 * np.pi * np.arange(200) / 20))


@pytest.fixture
def trend_signal() -> pd.Series:
    """Fixture for a signal with linear trend."""
    t = np.arange(100)
    return pd.Series(np.sin(2 * np.pi * t / 10) + 0.05 * t)


@pytest.mark.parametrize(
    "input_series, detrend_method, expected_period",
    [
        ("sine_wave_10", None, 10),
        ("sine_wave_20", None, 20),
        ("trend_signal", "linear", 10),
        ("trend_signal", [1], 10),
        ("constant", None, float("inf")),
    ],
    ids=["sine_wave_10", "sine_wave_20", "trend_signal", "trend_signal", "constant"],
)
def test_fft_analysis(
    request: pytest.FixtureRequest,
    input_series: str,
    detrend_method: Detrend,
    expected_period: int,
) -> None:
    """Test FFT analysis construction with various input signals.

    Args:
        request: Pytest fixture request to dynamically access fixtures.
        input_series: Name of the fixture providing the time series data.
        detrend_method: Method used for detrending or None.
        expected_period: Expected dominant period after FFT analysis.
    """
    series: pd.Series = request.getfixturevalue(input_series)
    fft = FFTAnalysis.from_series(series, detrend=detrend_method)

    # Test attributes and properties
    assert fft.dominant_freq == fft.frequencies[np.argmax(fft.fft_values)]
    assert fft.periods[0] == 0
    assert np.allclose(fft.frequencies[1:], 1 / fft.periods[1:])
    assert np.isclose(fft.dominant_period, expected_period, 0.5)
    assert fft.max_amplitude == fft.fft_values.max()
    assert fft.peak.equals(
        pd.DataFrame(
            {
                "frequency": [fft.frequencies[np.argmax(fft.fft_values)]],
                "period": [fft.dominant_period],
                "amplitude": [fft.fft_values.max()],
            }
        )
    )

    # Test get_spectrum method
    spectrum = fft.get_spectrum(max_freq=0.15)
    assert isinstance(spectrum, pd.DataFrame)
    assert list(spectrum.columns) == ["frequency", "period", "amplitude"]
    assert np.isclose(spectrum["frequency"].max(), 0.15, atol=0.01)
    assert (spectrum["frequency"] >= 0).all()
    assert spectrum["frequency"].is_monotonic_increasing

    # Test get_peaks method
    peaks = fft.get_peaks(max_freq=0.15, threshold=0.1)
    assert isinstance(peaks, pd.DataFrame)
    assert list(peaks.columns) == ["frequency", "period", "amplitude"]
    assert (peaks["amplitude"] > 0.1 * fft.max_amplitude).all()
