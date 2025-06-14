from collections.abc import Iterable
from typing import Any, Literal, Protocol, Self

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray

Detrend = Literal["linear"] | Iterable[int] | None


class FrequencyAnalysis(Protocol):  # pragma: no cover
    """Protocol defining the interface for frequency analysis of time series data.

    Any class implementing this protocol should provide methods for analyzing
    frequency components of time series data, identifying dominant frequencies,
    periods, and amplitudes.
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
        """Create a frequency analyzer instance from a pandas Series.

        Args:
            series: Time series data to analyze.
            detrend: Detrending method to apply before analysis:
                - "linear": Remove linear trend.
                - Iterable[int]: Apply differencing at specified lags.
                - None: No detrending.

        Returns:
            A frequency analyzer instance containing the analysis results.
        """
        ...

    @property
    def periods(self) -> NDArray[np.floating[Any]]:
        """Calculate the periods corresponding to each frequency.

        Returns:
            Array of period values (1/frequency).
        """
        ...

    @property
    def dominant_period(self) -> float:
        """Get the dominant period in the time series.

        Returns:
            Float representing the dominant period. For zero frequency (DC component),
            returns infinity to indicate a constant signal.
        """
        ...

    @property
    def max_amplitude(self) -> np.floating:
        """Get the maximum amplitude in the frequency spectrum.

        Returns:
            Maximum amplitude value.
        """
        ...

    @property
    def peak(self) -> pd.DataFrame:
        """Get a DataFrame with information about the dominant peak.

        Returns:
            DataFrame with frequency, period, and amplitude of the dominant peak.
        """
        ...

    def get_spectrum(self, max_freq: float = 0.15) -> pd.DataFrame:
        """Get the frequency spectrum up to a maximum frequency.

        Args:
            max_freq: Maximum frequency to include.

        Returns:
            DataFrame with frequency, period, and amplitude columns.
        """
        ...

    def get_peaks(self, max_freq: float = 0.5, threshold: float = 0.5) -> pd.DataFrame:
        """Get peaks in the frequency spectrum above a threshold.

        Args:
            max_freq: Maximum frequency to include.
            threshold: Threshold as a fraction of the maximum amplitude.

        Returns:
            DataFrame with frequency, period, and amplitude of peaks above threshold.
        """
        ...
