from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, Literal, Self

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray

from time_series_analysis.shared.interfaces import FrequencyAnalysis


@dataclass
class FFTAnalysis(FrequencyAnalysis):
    """Fast Fourier Transform analysis for time series data.

    This class performs FFT analysis on time series data, identifying dominant
    frequencies, periods, and amplitudes in the data.

    Attributes:
        fft_values: Array of FFT magnitudes.
        frequencies: Array of frequency values corresponding to fft_values.
        dominant_freq: The dominant frequency value.
    """

    fft_values: NDArray[np.floating[Any]]
    frequencies: NDArray[np.floating[Any]]
    dominant_freq: float

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        detrend: Literal["linear"] | Iterable[int] | None = "linear",
    ) -> Self:
        """Create an FFTAnalysis instance from a pandas Series.

        Performs FFT analysis on the input time series after optional detrending.

        Args:
            series: Time series data to analyze.
            detrend: Detrending method to apply before FFT:
                - "linear": Remove linear trend.
                - Iterable[int]: Apply differencing at specified lags.
                - None: No detrending.

        Returns:
            An FFTAnalysis instance containing the FFT results.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> # Create a simple sine wave with period=10
            >>> ts = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))
            >>> fft = FFTAnalysis.from_series(ts)
            >>> fft.dominant_period
            10.0
            >>> # Create a series with linear trend
            >>> trend_ts = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10) + 0.05 * np.arange(100))
            >>> fft_detrended = FFTAnalysis.from_series(trend_ts, detrend="linear")
            >>> fft_detrended.dominant_period
            10.0
        """
        lenght = len(series)
        if detrend is not None:
            if detrend == "linear":
                trend = np.polyfit(np.arange(len(series)), series.values, 1)
                series = series.values - np.polyval(trend, np.arange(lenght))
            elif isinstance(detrend, Iterable):
                for i in detrend:
                    series = series.diff(i)
                series = series.dropna()
                lenght = len(series)

        fft_values = np.abs(np.fft.rfft(series))
        frequencies = np.fft.rfftfreq(lenght)
        dominant_freq = float(frequencies[np.argmax(fft_values)])

        return cls(fft_values, frequencies, dominant_freq)

    @cached_property
    def periods(self) -> NDArray[np.floating[Any]]:
        """Calculate the periods corresponding to each frequency.

        Returns:
            Array of period values (1/frequency), with 0 for DC component (zero frequency).
            Using 0 is a convention to represent the infinite period of a constant signal.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> ts = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))
            >>> fft = FFTAnalysis.from_series(ts)
            >>> # Check if period 10 exists in the periods array
            >>> np.isclose(10, fft.periods).any()
            np.True_
            >>> # Check that DC component has period 0
            >>> fft_const = FFTAnalysis.from_series(pd.Series([1.0] * 100))
            >>> float(fft_const.periods[0])
            0.0
        """
        period = 1 / np.where(self.frequencies == 0, 1, self.frequencies)
        return np.where(self.frequencies == 0, 0, period)

    @cached_property
    def dominant_period(self) -> float:
        """Get the dominant period in the time series.

        Returns:
            Integer representing the dominant period. For zero frequency (DC component),
            returns 0 to indicate an infinite period or constant signal.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> # Create sine wave with period=20
            >>> ts = pd.Series(np.sin(2 * np.pi * np.arange(200) / 20))
            >>> fft = FFTAnalysis.from_series(ts)
            >>> fft.dominant_period
            20.0
            >>> # For constant time series, dominant frequency is 0, so period is represented as 0
            >>> constant_ts = pd.Series([5.0] * 100)
            >>> fft_constant = FFTAnalysis.from_series(constant_ts)
            >>> fft_constant.dominant_freq
            0.0
            >>> fft_constant.dominant_period
            inf
        """

        if self.dominant_freq > 0:
            return float(1.0 / self.dominant_freq)
        else:
            return float("inf")

    @cached_property
    def max_amplitude(self) -> np.floating:
        """Get the maximum amplitude in the FFT.

        Returns:
            Maximum amplitude value.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> # Create sine wave with amplitude 2
            >>> ts = pd.Series(2 * np.sin(2 * np.pi * np.arange(100) / 10))
            >>> fft = FFTAnalysis.from_series(ts)
            >>> # The FFT amplitude for a sine wave of amplitude 2 should be close to 100
            >>> bool(np.isclose(fft.max_amplitude, 100, rtol=0.1))
            True
        """
        return self.fft_values.max()

    @cached_property
    def peak(self) -> pd.DataFrame:
        """Get a DataFrame with information about the dominant peak.

        Returns:
            DataFrame with frequency, period, and amplitude of the dominant peak.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> ts = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))
            >>> fft = FFTAnalysis.from_series(ts)
            >>> peak_df = fft.peak
            >>> peak_df.shape
            (1, 3)
            >>> list(peak_df.columns)
            ['frequency', 'period', 'amplitude']
            >>> int(peak_df['period'].iloc[0])
            10
        """
        return pd.DataFrame(
            {
                "frequency": [self.dominant_freq],
                "period": [self.dominant_period],
                "amplitude": [self.max_amplitude],
            }
        )

    def get_spectrum(self, max_freq: float = 0.15) -> pd.DataFrame:
        """Get the frequency spectrum up to a maximum frequency.

        Args:
            max_freq: Maximum frequency to include (default: 0.15).

        Returns:
            DataFrame with frequency, period, and amplitude columns.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> ts = pd.Series(np.sin(2 * np.pi * np.arange(100) / 10))
            >>> fft = FFTAnalysis.from_series(ts)
            >>> spectrum = fft.get_spectrum(max_freq=0.2)
            >>> list(spectrum.columns)
            ['frequency', 'period', 'amplitude']
            >>> bool(spectrum['frequency'].max() <= 0.2)
            True
        """
        max_idx = np.searchsorted(self.frequencies, max_freq).item()

        frequency = self.frequencies[: max_idx + 1]
        period = self.periods[: max_idx + 1]
        amplitude = self.fft_values[: max_idx + 1]

        return pd.DataFrame(
            {
                "frequency": frequency,
                "period": period,
                "amplitude": amplitude,
            }
        )

    def get_peaks(self, max_freq: float = 0.5, threshold: float = 0.5) -> pd.DataFrame:
        """Get peaks in the frequency spectrum above a threshold.

        Args:
            max_freq: Maximum frequency to include (default: 0.15).
            threshold: Threshold as a fraction of the maximum amplitude (default: 0.5).

        Returns:
            DataFrame with frequency, period, and amplitude of peaks above threshold.
        """
        data = self.get_spectrum(max_freq)
        return data[data["amplitude"] > self.max_amplitude * threshold]
