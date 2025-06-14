from functools import cached_property
from typing import Iterable, Literal

import altair as alt
import numpy as np
import pandas as pd  # type: ignore
from altair.utils.display import MimeBundleType
from numpy.typing import NDArray
from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore

from .fourier import FFTAnalysis  # type: ignore
from .shared.interfaces import Detrend, FrequencyAnalysis
from .utils import ensure_iterable

FrequencyAnalysisCache = dict[str | tuple[int, ...], FrequencyAnalysis]


class TimeSeriesAnalyzer:
    """Comprehensive analyzer for time series data with visualization and frequency analysis.

    Provides a complete toolkit for analyzing time series data including FFT analysis,
    seasonal decomposition, stationarity testing, trend analysis, and interactive
    visualizations using Altair charts. Designed for exploratory data analysis,
    forecasting preparation, and pattern detection in temporal data.

    The analyzer supports out-of-time (OOT) validation by allowing you to mark
    periods at the end of the series for testing. These periods are excluded from
    analysis but highlighted in visualizations for model validation purposes.

    Attributes:
        df: DataFrame containing the complete time series data.
        y: Column name for the dependent variable (time series values).
        ds: Column name for the temporal dimension (dates/timestamps).
        unique_id: Column name for series identifier (useful for panel data).
        _y_name: Display name for y-axis in visualizations.
        _ds_name: Display name for time axis in visualizations.
        _unique_id_name: Display name for series identifier.
        _oot_periods: Number of periods reserved for out-of-time validation.
        _width: Default width for visualization charts in pixels.
        _height: Default height for visualization charts in pixels.
        _frequency_analysis_cache: Cache for expensive frequency analysis computations.

    Examples:
        Basic initialization with synthetic daily data:

        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2025-01-01', periods=100, freq='D')
        >>> values = np.sin(2 * np.pi * np.arange(100) / 7) + np.random.normal(0, 0.1, 100)
        >>> df = pd.DataFrame({'date': dates, 'value': values})
        >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
        >>> tsa.train_size
        100

        Configuration with custom display names and chart dimensions:

        >>> tsa_custom = TimeSeriesAnalyzer(
        ...     df, y='value', ds='date',
        ...     y_name='Revenue ($)', ds_name='Date',
        ...     width=800, height=500
        ... )
        >>> tsa_custom._y_name
        'Revenue ($)'

        Setup for model validation with out-of-time periods:

        >>> tsa_oot = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=14)
        >>> tsa_oot.train_size
        86
        >>> tsa_oot.oot_periods
        14

        Panel data with multiple series:

        >>> panel_df = pd.concat([
        ...     df.assign(series_id='A'),
        ...     df.assign(series_id='B')
        ... ], ignore_index=True)
        >>> panel_tsa = TimeSeriesAnalyzer(
        ...     panel_df, y='value', ds='date', unique_id='series_id'
        ... )
        >>> 'series_id' in panel_tsa.df.columns
        True
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str = "y",
        ds: str = "ds",
        unique_id: str = "unique_id",
        y_name: str | None = None,
        ds_name: str | None = None,
        unique_id_name: str | None = None,
        width: int = 600,
        height: int = 400,
        oot_periods: int = 0,
        frequency_analysis_tool: type[FrequencyAnalysis] = FFTAnalysis,
    ) -> None:
        """Initialize the TimeSeriesAnalyzer with data and configuration options.

        Args:
            df: DataFrame containing time series data with at minimum temporal
                and value columns.
            y: Column name for the dependent variable (time series values).
                Default is 'y' following Nixtla convention.
            ds: Column name for the temporal dimension (dates/timestamps).
                Default is 'ds' following Nixtla convention.
            unique_id: Column name for series identifier in panel data scenarios.
                Default is 'unique_id' following Nixtla convention.
            y_name: Human-readable label for the y-axis in visualizations.
                If None, uses the y column name.
            ds_name: Human-readable label for the time axis in visualizations.
                If None, uses the ds column name.
            unique_id_name: Human-readable label for series identifiers.
                If None, uses the unique_id column name.
            width: Default width for visualization charts in pixels.
            height: Default height for visualization charts in pixels.
            oot_periods: Number of periods at the end of the series to reserve
                for out-of-time validation. These periods are excluded from
                analysis and visualizations.
            frequency_analysis_tool: Class to use for frequency domain analysis.
                Default is FFTAnalysis, but can be customized for different
                analysis approaches.

        Examples:
            Minimal setup with default column names:

            >>> df = pd.DataFrame({
            ...     'ds': pd.date_range('2025-01-01', periods=50),
            ...     'y': np.random.randn(50)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df)
            >>> tsa.y
            'y'
            >>> tsa.ds
            'ds'

            Custom column names and labels:

            >>> sales_df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=30),
            ...     'revenue': np.random.uniform(1000, 2000, 30)
            ... })
            >>> tsa = TimeSeriesAnalyzer(
            ...     sales_df, y='revenue', ds='date',
            ...     y_name='Daily Revenue ($)', ds_name='Business Date'
            ... )
            >>> tsa._y_name
            'Daily Revenue ($)'

            Configuration for large dashboard displays:

            >>> tsa_large = TimeSeriesAnalyzer(
            ...     df, width=1200, height=800
            ... )
            >>> tsa_large._width
            1200
        """
        self.df = df
        self.unique_id = unique_id
        self.ds = ds
        self.y = y

        self._y_name = y_name or y
        self._ds_name = ds_name or ds
        self._unique_id_name = unique_id_name or unique_id
        self._oot_periods = oot_periods

        self._width = width
        self._height = height

        self._var_name = "Series"
        self._value_name = "Values"

        self._frequency_analysis_cache: FrequencyAnalysisCache = {}
        self._frequency_analysis = frequency_analysis_tool

    @property
    def oot_periods(self) -> int:
        """Get the number of out-of-time periods reserved for validation.

        Returns:
            Number of periods at the end of the time series marked as out-of-time.
            These periods are excluded from analysis and visualizations.

        Examples:
            Check current out-of-time configuration:

            >>> df = pd.DataFrame({
            ...     'ds': pd.date_range('2025-01-01', periods=100),
            ...     'y': np.random.randn(100)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, oot_periods=10)
            >>> tsa.oot_periods
            10
        """
        return self._oot_periods

    @oot_periods.setter
    def oot_periods(self, value: int) -> None:
        """Set the number of out-of-time periods and invalidate cached analyses.

        Changing the out-of-time periods affects the training data size and
        therefore invalidates all cached computations and properties.

        Args:
            value: Number of periods at the end of the series to mark as out-of-time.
                Must be non-negative and less than the total series length.

        Examples:
            Update out-of-time periods and observe cache invalidation:

            >>> df = pd.DataFrame({
            ...     'ds': pd.date_range('2025-01-01', periods=100),
            ...     'y': np.random.randn(100)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, oot_periods=5)
            >>> initial_size = tsa.train_size
            >>> tsa.oot_periods = 15
            >>> tsa.train_size == initial_size - 15
            True
        """
        self._oot_periods = value
        self._invalidate_cache()

    @cached_property
    def train_data(self) -> pd.DataFrame:
        """Get the training dataset excluding out-of-time periods.

        Returns:
            DataFrame with standardized column names ('unique_id', 'ds', 'y')
            containing only the training portion of the data.

        Examples:
            Access training data with standardized column names:

            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=100),
            ...     'value': np.random.randn(100),
            ...     'series': ['A'] * 100
            ... })
            >>> tsa = TimeSeriesAnalyzer(
            ...     df, y='value', ds='date', unique_id='series', oot_periods=10
            ... )
            >>> train = tsa.train_data
            >>> len(train)
            90
            >>> list(train.columns)
            ['ds', 'y', 'unique_id']
        """
        columns = {self.unique_id: "unique_id", self.ds: "ds", self.y: "y"}
        df = self.df
        if self.unique_id not in df.columns:
            df = df.assign(unique_id=1)
        return df[: len(df) - self.oot_periods].rename(columns=columns)

    @cached_property
    def oot_data(self) -> pd.DataFrame:
        """Get the out-of-time validation dataset.

        Returns:
            DataFrame with standardized column names ('unique_id', 'ds', 'y')
            containing only the out-of-time portion of the data.

        Examples:
            Access out-of-time validation data:

            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=100),
            ...     'value': np.random.randn(100)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=15)
            >>> oot = tsa.oot_data
            >>> len(oot)
            15

            When no out-of-time periods are set:

            >>> tsa_no_oot = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> len(tsa_no_oot.oot_data)
            0
        """
        columns = {self.unique_id: "unique_id", self.ds: "ds", self.y: "y"}
        df = self.df
        if self.unique_id not in df.columns:
            df = df.assign(unique_id=1)
        return df[len(df) - self.oot_periods :].rename(columns=columns)

    @cached_property
    def dates(self) -> pd.Series:
        """Get the temporal index for training data.

        Returns:
            Series containing the dates/timestamps used for in-time analysis,
            excluding any out-of-time periods.

        Examples:
            Access training period dates:

            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=30),
            ...     'value': np.random.randn(30)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=5)
            >>> dates = tsa.dates
            >>> len(dates)
            25
            >>> dates.dtype
            dtype('<M8[ns]')
        """
        return self.df[self.ds][: len(self.df) - self.oot_periods]

    @cached_property
    def series(self) -> pd.Series:
        """Get the time series values for training data.

        Returns:
            Series containing the dependent variable values used for analysis,
            excluding any out-of-time periods.

        Examples:
            Access training period values:

            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=50),
            ...     'value': np.arange(50) + np.random.normal(0, 0.1, 50)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=10)
            >>> series = tsa.series
            >>> len(series)
            40
            >>> series.dtype
            dtype('float64')
        """
        return self.df[self.y][: len(self.df) - self.oot_periods]

    @cached_property
    def train_size(self) -> int:
        """Get the number of observations in the training dataset.

        Returns:
            Integer representing the size of the training data after excluding
            out-of-time periods.

        Examples:
            Check training data size:

            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=365),
            ...     'value': np.random.randn(365)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=30)
            >>> tsa.train_size
            335

            Relationship with total data size:

            >>> total_size = len(tsa.df)
            >>> tsa.train_size == total_size - tsa.oot_periods
            True
        """
        return len(self.train_data)

    def plot_trends(
        self,
        polyfit_degrees: int | Iterable[int] | None = None,
        moving_average_windows: int | Iterable[int] | None = None,
    ) -> alt.LayerChart | alt.VConcatChart:
        """Plot trend analysis with interactive highlighting for trend components.

        Creates interactive visualizations showing the original time series overlaid
        with trend components. Supports polynomial trend fitting of various degrees
        and moving averages with different window sizes for trend identification.

        Args:
            polyfit_degrees: Polynomial degree(s) for trend fitting. Can be a single
                integer or iterable of integers.
            moving_average_windows: Window size(s) for moving average calculation.
                Can be a single integer or iterable of integers.

        Returns:
            Altair chart with interactive highlighting. Returns LayerChart for single
            trend type, VConcatChart when both trend types are requested.

        Raises:
            ValueError: If both polyfit_degrees and moving_average_windows are None.

        Examples:
            Interactive polynomial trend analysis:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=200)
            >>> values = (0.01 * np.arange(200)**2 +
            ...          np.sin(2 * np.pi * np.arange(200) / 30) +
            ...          np.random.normal(0, 0.5, 200))
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> chart = tsa.plot_trends(polyfit_degrees=[1, 2, 3])
            >>> chart.title
            'Polynomial Fits'

            Interactive moving averages with highlighting:

            >>> ma_chart = tsa.plot_trends(moving_average_windows=[5, 15, 30])
            >>> ma_chart.title
            'Moving Averages'

            Combined analysis with enhanced interactivity:

            >>> combined = tsa.plot_trends(
            ...     polyfit_degrees=[1, 2], moving_average_windows=[10, 20]
            ... )
            >>> combined.title
            'Trends'
        """
        if polyfit_degrees is None and moving_average_windows is None:
            msg = "At least one of polyfit_degrees or moving_average_windows must be provided"
            raise ValueError(msg)

        ma_chart: alt.LayerChart | None = None
        pf_chart: alt.LayerChart | None = None
        chart: alt.LayerChart | alt.VConcatChart

        series_data = (
            self.dates.to_frame(self._ds_name)
            .join(self.series.to_frame(self._y_name))
            .melt(
                id_vars=self._ds_name,
                var_name=self._var_name,
                value_name=self._value_name,
            )
        )
        ts_line = alt.Chart(series_data).mark_line()

        if polyfit_degrees:
            pf_df = self._compute_polynomial_fit(polyfit_degrees)
            pf_line = self._highlight(alt.Chart(pf_df).mark_line())
            title = "Polynomial Fits"

            pf_chart = (
                alt.layer(ts_line, pf_line)
                .encode(
                    x=alt.X(f"{self._ds_name}:T"),
                    y=alt.Y(f"{self._value_name}:Q"),
                    color=alt.Color(f"{self._var_name}:N"),
                )
                .properties(width=self._width, height=self._height, title=title)
            )

            if not moving_average_windows:
                chart = pf_chart

        if moving_average_windows:
            ma_df = self._compute_moving_averages(moving_average_windows)
            ma_line = self._highlight(alt.Chart(ma_df).mark_line())
            title = "Moving Averages"

            ma_chart = (
                alt.layer(ts_line, ma_line)
                .encode(
                    x=alt.X(f"{self._ds_name}:T"),
                    y=alt.Y(f"{self._value_name}:Q"),
                    color=alt.Color(f"{self._var_name}:N"),
                )
                .properties(width=self._width, height=self._height, title=title)
            )

            if not polyfit_degrees:
                chart = ma_chart

        if ma_chart and pf_chart:
            title = "Trends"
            chart = alt.vconcat(
                ma_chart.properties(width=self._width, height=self._height // 2),
                pf_chart.properties(width=self._width, height=self._height // 2),
            ).properties(title=title)

        return chart

    def plot_fft_analysis(
        self, max_freq: float = 0.5, threshold: float = 0.5, detrend: Detrend = None
    ) -> alt.VConcatChart:
        """Plot the frequency spectrum of the time series using FFT.

        Creates a visualization with two charts:
        1. Frequency Spectrum - showing amplitude vs frequency
        2. Period Spectrum - showing amplitude vs period (1/frequency)

        The visualization highlights peak frequencies that exceed the specified
        threshold, indicates the dominant frequency/period, and marks the DC
        component (zero frequency) with a dashed line in the period chart.

        Args:
            max_freq: Maximum frequency to display (default: 0.5)
                This can be used to limit the x-axis range for clearer visualization.
                The value of 0.5 is the Nyquist frequency.
            threshold: Minimum amplitude threshold for peak detection (default: 0.5)
                Expressed as a fraction of the maximum amplitude.
            detrend: Method to detrend the time series before FFT (default: None)
                Can be None, "linear", or a sequence of integers for differencing.
                Detrending helps to better identify seasonal patterns by removing
                trends that might dominate the frequency spectrum.

        Returns:
            alt.VConcatChart: An Altair chart with frequency and period spectrums

        Examples:
            Standard FFT analysis of synthetic weekly pattern:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=365, freq='D')
            >>> values = np.sin(2 * np.pi * np.arange(365) / 7) + np.random.normal(0, 0.1, 365)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, 'value', 'date')
            >>> chart = tsa.plot_fft_analysis()
            >>> chart.title.startswith('FFT Analysis')
            True

            Zooming in on lower frequencies for longer periods:

            >>> low_freq_chart = tsa.plot_fft_analysis(max_freq=0.2)
            >>> 'FFT Analysis' in low_freq_chart.title
            True

            Using lower threshold to detect more peaks:

            >>> sensitive_chart = tsa.plot_fft_analysis(threshold=0.3)
            >>> 'Dominant Period' in sensitive_chart.title
            True

            Analysis with linear detrending to remove trends:

            >>> trend_values = values + 0.05 * np.arange(365)
            >>> trend_df = pd.DataFrame({'date': dates, 'value': trend_values})
            >>> trend_tsa = TimeSeriesAnalyzer(trend_df, 'value', 'date')
            >>> detrended_chart = trend_tsa.plot_fft_analysis(detrend="linear")
            >>> 'Detrended: linear' in detrended_chart.title
            True

            Analysis with differencing to remove trends:

            >>> diff_chart = trend_tsa.plot_fft_analysis(detrend=[1])
            >>> 'Differenced: [1]' in diff_chart.title
            True
        """
        fft_analysis = self._get_or_create_fft_analysis(detrend=detrend)

        frequency_spectrum, peak_frequencies, annotation_data = (
            fft_analysis.get_spectrum(max_freq).assign(series_type="Spectrum"),
            fft_analysis.get_peaks(max_freq, threshold).assign(series_type="Peak"),
            fft_analysis.peak,
        )

        dc_value = {"amplitude": [frequency_spectrum.iloc[0]["amplitude"]]}
        dc = pd.DataFrame(dc_value).assign(series_type="DC Component")

        dominant_period = str(round(fft_analysis.dominant_period, 2))
        if dominant_period == "inf":
            dominant_period = "DC (∞)"
        dominant_period_text = f"Dominant Period: {dominant_period}"

        x_frequency = alt.X("frequency:Q", title="Frequency")
        x_period = alt.X(
            "period:Q",
            title="Period",
            scale=alt.Scale(type="log", domain=[2, frequency_spectrum["period"].max()]),
        )
        y_amplitude = alt.Y("amplitude:Q", title="Amplitude")
        color = alt.Color(
            "series_type:N",
            legend=alt.Legend(title="Components"),
            scale=alt.Scale(
                domain=["Spectrum", "Peak", "DC Component"],
                range=["#1f77b4", "red", "#333333"],
            ),
        )
        tooltip = [
            alt.Tooltip("frequency:Q"),
            alt.Tooltip("period:Q"),
            alt.Tooltip("amplitude:Q"),
        ]
        text = alt.value(dominant_period_text)

        base_spectrum = (
            alt.Chart(frequency_spectrum)
            .mark_line()
            .encode(y=y_amplitude, color=color)
            .properties(width=self._width, height=self._height // 2)
        )

        base_peaks = (
            alt.Chart(peak_frequencies)
            .mark_point(color="red", size=100)
            .encode(y=y_amplitude, color=color, tooltip=tooltip)
        )

        base_annotation = (
            alt.Chart(annotation_data)
            .mark_text(align="left", baseline="middle", fontSize=12, dx=5, dy=-10)
            .encode(y=y_amplitude, text=text, tooltip=tooltip)
        )

        dc_line = (
            alt.Chart(dc)
            .mark_rule(strokeDash=[10, 5])
            .encode(y=y_amplitude, color=color)
        )

        frequency = alt.layer(
            base_spectrum.encode(x=x_frequency),
            base_peaks.encode(x=x_frequency),
            base_annotation.encode(x=x_frequency),
        )

        period = alt.layer(
            base_spectrum.transform_filter(alt.datum.period > 0).encode(x=x_period),
            base_peaks.transform_filter(alt.datum.period > 0).encode(x=x_period),
            base_annotation.transform_filter(alt.datum.period > 0).encode(x=x_period),
            dc_line,
        )

        detrend_text = ""
        if detrend is not None:
            if isinstance(detrend, str):
                detrend_text = f" (Detrended: {detrend})"
            else:
                detrend_text = f" (Differenced: {detrend})"

        return alt.vconcat(
            frequency.properties(title="Frequency Spectrum"),
            period.properties(title="Period Spectrum"),
        ).properties(title=f"FFT Analysis{detrend_text} - {dominant_period_text}")

    def plot_diffs(self, *differences: int | Iterable[int]) -> alt.LayerChart:
        """Plot difference charts for multiple series with interactive highlighting.

        Creates an interactive line chart visualization that displays differences
        between consecutive observations at various lags. This is useful for
        identifying stationarity, seasonal patterns, and appropriate differencing
        strategies for time series modeling.

        Differencing helps remove trends and seasonal patterns to achieve stationarity,
        which is required for many time series analysis techniques. The interactive
        highlighting makes it easy to compare different differencing strategies.

        Args:
            *differences: One or more integers or iterables of integers representing
                the differencing lags to plot. Each difference creates a separate
                line in the visualization. Examples:
                - differences(1): First difference (y[t] - y[t-1])
                - differences(7): Seventh difference for weekly data
                - differences([1, 7]): Combined first and seventh differencing
                - differences(1, 7, [1, 7]): Three separate lines

        Returns:
            alt.LayerChart: An Altair layered chart with interactive highlighting,
                sized according to the instance's width and height settings.

        Examples:
            Single differencing for trend removal:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=100)
            >>> trend = 0.1 * np.arange(100)
            >>> values = trend + np.random.normal(0, 0.5, 100)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> chart = tsa.plot_diffs(1)
            >>> chart.title
            'Differences'

            Multiple lags for seasonal pattern analysis:

            >>> seasonal_values = (trend +
            ...                   np.sin(2 * np.pi * np.arange(100) / 7) +
            ...                   np.random.normal(0, 0.3, 100))
            >>> seasonal_df = pd.DataFrame({'date': dates, 'value': seasonal_values})
            >>> seasonal_tsa = TimeSeriesAnalyzer(seasonal_df, y='value', ds='date')
            >>> multi_chart = seasonal_tsa.plot_diffs(1, 7, 14)
            >>> 'Differences' in multi_chart.title
            True

            Combined differencing strategies:

            >>> combined_chart = seasonal_tsa.plot_diffs([1, 7])
            >>> 'Diffs 1, 7' in str(combined_chart.data)
            True

            Comparison of different approaches:

            >>> comparison = seasonal_tsa.plot_diffs(1, 7, [1, 7])
            >>> len(comparison.layer) >= 2
            True
        """

        data = self._compute_differences(*differences)
        base = alt.Chart(data).encode(
            x=alt.X(f"{self._ds_name}:T"),
            y=alt.Y(f"{self._value_name}:Q"),
            color=alt.Color(f"{self._var_name}:N"),
        )

        chart = self._highlight(base, self._var_name)
        title = "Differences"

        return chart.properties(width=self._width, height=self._height, title=title)

    def plot_seasonal_decomposition(
        self,
        period: int | None = None,
        detrend: Detrend = "linear",
        model: Literal["multiplicative", "additive"] = "additive",
    ) -> alt.VConcatChart:
        """Decompose the time series into trend, seasonal, and residual components.

        This method applies seasonal decomposition to break down the time series
        into its constituent components. If no period is provided, it automatically
        detects the dominant period using FFT analysis. The decomposition helps
        understand the underlying structure of the time series.

        Args:
            period: Number of observations per seasonal cycle.
                If None, will be automatically detected using FFT analysis.
            detrend: Method to detrend the series before period detection (if period is None).
                Only applies to the period detection step, not the decomposition itself.
            model: Type of seasonal component:
                - "additive": Y = trend + seasonal + residual
                - "multiplicative": Y = trend * seasonal * residual

        Returns:
            Altair chart showing the original series and its decomposed components:
                - Original time series
                - Trend component
                - Seasonal component
                - Residual component

        Notes:
            - For additive models, residuals are positive/negative deviations
            - For multiplicative models, residuals are percentage deviations
            - The method uses statsmodels.tsa.seasonal.seasonal_decompose internally
            - First and last observations might have NaN values for trend component

        Examples:
            Decomposition with auto-detected weekly seasonality:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=365, freq='D')
            >>> trend = 0.05 * np.arange(365)
            >>> seasonal = np.sin(2 * np.pi * np.arange(365) / 7)
            >>> noise = np.random.normal(0, 0.1, 365)
            >>> values = trend + seasonal + noise
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> chart = tsa.plot_seasonal_decomposition(detrend="linear")
            >>> 'Seasonality Decomposition' in chart.title
            True

            Decomposition with manually specified period:

            >>> manual_chart = tsa.plot_seasonal_decomposition(period=7)
            >>> 'Period=7' in manual_chart.title
            True

            Multiplicative decomposition:

            >>> mult_values = (1 + trend/10) * (1 + seasonal/2) * (1 + noise/10)
            >>> mult_df = pd.DataFrame({'date': dates, 'value': mult_values})
            >>> mult_tsa = TimeSeriesAnalyzer(mult_df, y='value', ds='date')
            >>> mult_chart = mult_tsa.plot_seasonal_decomposition(model="multiplicative")
            >>> 'Model=multiplicative' in mult_chart.title
            True

            Monthly seasonality detection:

            >>> monthly_dates = pd.date_range('2020-01-01', periods=1000, freq='D')
            >>> monthly_seasonal = np.sin(2 * np.pi * np.arange(1000) / 30)
            >>> monthly_df = pd.DataFrame({
            ...     'date': monthly_dates,
            ...     'value': monthly_seasonal + np.random.normal(0, 0.1, 1000)
            ... })
            >>> monthly_tsa = TimeSeriesAnalyzer(monthly_df, y='value', ds='date')
            >>> monthly_chart = monthly_tsa.plot_seasonal_decomposition()
            >>> 'Period=30' in monthly_chart.title
            True
        """
        if period is None:
            fft_period = self.find_period_fft(detrend=detrend)
            period = 1 if fft_period == float("inf") else int(round(fft_period, 0))

        decomposed = seasonal_decompose(self.series, model=model, period=period)

        trend: pd.Series = decomposed.trend
        seasonal: pd.Series = decomposed.seasonal
        resid: pd.Series = decomposed.resid
        title = f"Seasonality Decomposition (Period={period}, Model={model})"

        other = [self.series, trend, seasonal, resid]
        data = self.dates.to_frame().join(other, how="left")

        base = (
            alt.Chart(data)
            .encode(x=alt.X(self.ds, title=self._ds_name))
            .properties(height=self._height // 8, width=self._width)
        )
        line, point = base.mark_line(), base.mark_point(size=1)

        return (
            line.encode(y=alt.Y(self.y, title=self._y_name).scale(zero=False))
            & line.encode(y=alt.Y("trend").scale(zero=False))
            & line.encode(y=alt.Y("seasonal").scale(zero=False))
            & point.encode(y=alt.Y("resid").scale(zero=False))
        ).properties(title=title)

    def list_peaks(
        self, threshold: float = 0.5, detrend: Detrend = None
    ) -> pd.DataFrame:
        """List significant frequency peaks in the time series spectrum.

        Identifies and returns frequency peaks that exceed the specified threshold
        in the FFT analysis. This is useful for discovering multiple seasonal
        patterns, harmonics, and other periodic components in the data.

        Args:
            threshold: Minimum amplitude threshold for peak detection as a fraction
                of the maximum amplitude (default: 0.5). Lower values detect more
                peaks but may include noise.
            detrend: Method to detrend the time series before FFT analysis:
                - None: No detrending (default)
                - "linear": Remove linear trend
                - Sequence of ints: Apply differencing at specified lags

        Returns:
            DataFrame with columns ['frequency', 'period', 'amplitude'] containing
            information about detected peaks, sorted by amplitude in descending order.

        Examples:
            Detect major peaks in synthetic data with multiple periodicities:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=365)
            >>> weekly = np.sin(2 * np.pi * np.arange(365) / 7)
            >>> monthly = 0.5 * np.sin(2 * np.pi * np.arange(365) / 30)
            >>> values = weekly + monthly + np.random.normal(0, 0.1, 365)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> peaks = tsa.list_peaks(threshold=0.3)
            >>> len(peaks) >= 2
            True
            >>> 'frequency' in peaks.columns
            True

            Conservative peak detection with high threshold:

            >>> major_peaks = tsa.list_peaks(threshold=0.8)
            >>> len(major_peaks) <= len(peaks)
            True

            Peak detection with detrending:

            >>> trend_values = values + 0.01 * np.arange(365)
            >>> trend_df = pd.DataFrame({'date': dates, 'value': trend_values})
            >>> trend_tsa = TimeSeriesAnalyzer(trend_df, y='value', ds='date')
            >>> detrended_peaks = trend_tsa.list_peaks(detrend="linear")
            >>> len(detrended_peaks) >= 1
            True

            Access peak information:

            >>> if len(peaks) > 0:
            ...     dominant_period = peaks.iloc[0]['period']
            ...     dominant_amplitude = peaks.iloc[0]['amplitude']
            ...     isinstance(dominant_period, (int, float))
            ... else:
            ...     True
            True
        """
        fft_analysis = self._get_or_create_fft_analysis(detrend=detrend)
        return fft_analysis.get_peaks(threshold=threshold)

    def find_period_fft(self, detrend: Detrend = None) -> float:
        """Find the dominant period of the time series using Fast Fourier Transform.

        This method uses the frequency domain to identify the dominant period in the
        time series data. It can optionally detrend the data first to better detect
        periodic patterns obscured by trends.

        Args:
            detrend: Method to detrend the time series before FFT analysis:
                - None: No detrending (default)
                - "linear": Remove linear trend
                - Sequence of ints: Apply differencing at specified lags

        Returns:
            float: The estimated dominant period, rounded to 2 decimal places.
                For a constant signal (DC component), returns inf (infinity).

        Examples:
            Detect weekly seasonality in daily data:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=365, freq='D')
            >>> values = np.sin(2 * np.pi * np.arange(365) / 7) + np.random.normal(0, 0.1, 365)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> analyzer = TimeSeriesAnalyzer(df, 'value', 'date')
            >>> period = analyzer.find_period_fft()
            >>> bool(np.isclose(7, period, rtol=0.1))
            True

            Handle trending data with detrending:

            >>> trend_values = values + 0.05 * np.arange(365)
            >>> trend_df = pd.DataFrame({'date': dates, 'value': trend_values})
            >>> trend_analyzer = TimeSeriesAnalyzer(trend_df, 'value', 'date')
            >>> trend_analyzer.find_period_fft() == float('inf')
            True
            >>> period_detrended = trend_analyzer.find_period_fft(detrend="linear")
            >>> bool(np.isclose(7, period_detrended, rtol=0.1))
            True

            Detect monthly patterns in longer series:

            >>> monthly_dates = pd.date_range('2020-01-01', periods=1000, freq='D')
            >>> monthly_values = np.sin(2 * np.pi * np.arange(1000) / 30) + np.random.normal(0, 0.2, 1000)
            >>> monthly_df = pd.DataFrame({'date': monthly_dates, 'value': monthly_values})
            >>> monthly_analyzer = TimeSeriesAnalyzer(monthly_df, 'value', 'date')
            >>> monthly_period = monthly_analyzer.find_period_fft()
            >>> bool(np.isclose(30, monthly_period, rtol=0.15))
            True

            Constant signal returns infinite period:

            >>> const_df = pd.DataFrame({'date': dates, 'value': [5] * 365})
            >>> const_analyzer = TimeSeriesAnalyzer(const_df, 'value', 'date')
            >>> const_analyzer.find_period_fft() == float('inf')
            True
        """
        fft_analysis = self._get_or_create_fft_analysis(detrend=detrend)
        return round(fft_analysis.dominant_period, 2)

    def adf_test(self, *diffs: int) -> pd.Series:
        """Perform the Augmented Dickey-Fuller test for stationarity assessment.

        The Augmented Dickey-Fuller test determines if a time series is stationary.
        A stationary time series has constant statistical properties (mean, variance)
        over time, which is important for many time series analysis techniques.

        This implementation allows you to apply differencing to the time series
        before performing the test. Differencing can help transform a non-stationary
        series into a stationary one by removing trends and seasonal patterns.

        Args:
            *diffs: Zero or more integers representing differencing orders to apply
                sequentially before the ADF test. Examples:
                - No args: Test original series
                - diffs(1): Test first difference (y[t] - y[t-1])
                - diffs(1, 12): Apply first difference, then 12th difference

        Returns:
            pd.Series: Series containing ADF test results with the following index:
                - 'Test Statistic': ADF test statistic value
                - 'p-value': P-value for the test
                - 'Critical Value (1%)', (5%)', (10%): Critical values at different
                  significance levels

        Notes:
            The null hypothesis is that the time series is non-stationary.
            A small p-value (typically < 0.05) suggests the series is stationary.
            More negative test statistics provide stronger evidence for stationarity.

        Examples:
            Test stationarity of trending time series:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=100, freq='D')
            >>> values = np.arange(100) + np.random.normal(0, 5, 100)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> results = tsa.adf_test()
            >>> 'Test Statistic' in results.index
            True
            >>> 'p-value' in results.index
            True

            Test should indicate non-stationarity for trending data:

            >>> results['p-value'] > 0.05
            np.True_

            Test after first differencing should show stationarity:

            >>> results_diff = tsa.adf_test(1)
            >>> results_diff['p-value'] < 0.05
            np.True_

            Test stationary white noise series:

            >>> white_noise = pd.DataFrame({
            ...     'date': dates,
            ...     'value': np.random.normal(0, 1, 100)
            ... })
            >>> noise_tsa = TimeSeriesAnalyzer(white_noise, y='value', ds='date')
            >>> noise_results = noise_tsa.adf_test()
            >>> noise_results['p-value'] < 0.05
            np.True_

            Multiple differencing for seasonal data:

            >>> seasonal_trend = (np.arange(100) +
            ...                  np.sin(2 * np.pi * np.arange(100) / 12) +
            ...                  np.random.normal(0, 1, 100))
            >>> seasonal_df = pd.DataFrame({'date': dates, 'value': seasonal_trend})
            >>> seasonal_tsa = TimeSeriesAnalyzer(seasonal_df, y='value', ds='date')
            >>> seasonal_results = seasonal_tsa.adf_test(1, 12)
            >>> 'Critical Value (5%)' in seasonal_results.index
            True
        """

        series = self.series
        for diff in diffs:
            series = series.diff(diff)
        results = adfuller(series.dropna())

        data = results[0:2] + tuple(results[4].values())
        critical_values = ["Critical Value (%s)" % key for key in results[4].keys()]
        index = ["Test Statistic", "p-value", *critical_values]

        return pd.Series(data, index=index)

    def _get_or_create_fft_analysis(self, detrend: Detrend = None) -> FrequencyAnalysis:
        """Perform Fast Fourier Transform analysis with caching for efficiency.

        This method computes or retrieves a cached FFT analysis of the time series.
        Caching prevents recomputation when the same detrending method is used
        multiple times, improving performance for interactive analysis.

        Args:
            detrend: Method to detrend the time series before FFT analysis:
                - None: No detrending
                - "linear": Remove linear trend
                - Sequence of ints: Apply differencing at specified lags

        Returns:
            FrequencyAnalysis object containing the results of the frequency analysis
            with methods for accessing spectrum, peaks, and dominant frequencies.

        Examples:
            Cache utilization for repeated analyses:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=200)
            >>> values = np.sin(2 * np.pi * np.arange(200) / 10) + np.random.normal(0, 0.1, 200)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> analysis1 = tsa._get_or_create_fft_analysis(detrend="linear")
            >>> analysis2 = tsa._get_or_create_fft_analysis(detrend="linear")
            >>> analysis1 is analysis2
            True

            Different detrending methods create separate cache entries:

            >>> analysis_none = tsa._get_or_create_fft_analysis(detrend=None)
            >>> analysis_linear = tsa._get_or_create_fft_analysis(detrend="linear")
            >>> analysis_none is not analysis_linear
            True
        """
        key: str | tuple[int, ...]
        if detrend is None:
            key = "none"
        else:
            if isinstance(detrend, str):
                key = detrend
            else:
                key = tuple(detrend)

        fft_analysis = self._frequency_analysis_cache.get(key, None)
        if fft_analysis is None:
            fft_analysis = self._frequency_analysis.from_series(
                self.series, detrend=detrend
            )
            self._frequency_analysis_cache[key] = fft_analysis
        return fft_analysis

    def _highlight(self, chart: alt.Chart, field: str | None = None) -> alt.LayerChart:
        """Add interactive highlighting to Altair charts for better user experience.

        Creates an interactive chart layer that emphasizes lines when hovering over
        them. This improves visualization clarity when multiple series are displayed
        by making the selected series more prominent.

        Args:
            chart: Base Altair chart to add highlighting to.
            field: Field name to use for selection grouping. If None, uses the
                default series variable name.

        Returns:
            LayerChart with interactive highlighting that increases line width
            on hover and adds invisible hover targets for better interaction.

        Examples:
            Basic highlighting for multi-series chart:

            >>> import pandas as pd
            >>> import numpy as np
            >>> import altair as alt
            >>> data = pd.DataFrame({
            ...     'x': list(range(10)) * 2,
            ...     'y': np.random.randn(20),
            ...     'series': ['A'] * 10 + ['B'] * 10
            ... })
            >>> base_chart = alt.Chart(data).mark_line().encode(
            ...     x='x:Q', y='y:Q', color='series:N'
            ... )
            >>> tsa = TimeSeriesAnalyzer(pd.DataFrame({'ds': [1], 'y': [1]}))
            >>> highlighted = tsa._highlight(base_chart, 'series')
            >>> len(highlighted.layer)
            2
        """
        if field is None:
            field = self._var_name
        hover = alt.selection_point(on="pointerover", fields=[field], nearest=True)
        line_size = alt.when(hover).then(alt.value(3)).otherwise(alt.value(1))

        return alt.layer(
            chart.mark_circle().encode(opacity=alt.value(0)).add_params(hover),
            chart.mark_line().encode(size=line_size),
        )

    def _compute_differences(self, *differences: int | Iterable[int]) -> pd.DataFrame:
        """Compute differenced series for multiple differencing strategies.

        Applies various differencing operations to the time series and returns
        a melted DataFrame suitable for visualization. Each differencing strategy
        creates a separate series in the result.

        Args:
            *differences: One or more differencing specifications. Each can be:
                - Single integer: Apply that lag difference
                - Iterable of integers: Apply differences sequentially

        Returns:
            DataFrame in long format with columns for dates, series names, and values.
            Series names follow the pattern "Diffs X, Y, Z" where X, Y, Z are the
            applied difference lags.

        Examples:
            Single and multiple differencing computations:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=20)
            >>> values = np.arange(20) + np.random.normal(0, 0.1, 20)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> diffs = tsa._compute_differences(1, 7, [1, 7])
            >>> 'Diffs 1' in diffs[tsa._var_name].values
            True
            >>> 'Diffs 7' in diffs[tsa._var_name].values
            True
            >>> 'Diffs 1, 7' in diffs[tsa._var_name].values
            True
        """
        results: dict[str, pd.Series] = {}
        for diffs in differences:
            series = self.series.copy()
            diffs_ = ensure_iterable(diffs)
            for d in diffs_:
                series = series.diff(d)
            results[f"Diffs {', '.join(str(d) for d in diffs_)}"] = series

        return self._melt(results)

    def _compute_moving_averages(self, windows: int | Iterable[int]) -> pd.DataFrame:
        """Compute moving averages with different window sizes for trend analysis.

        Calculates centered moving averages using different window sizes to identify
        trends at various time scales. Returns a melted DataFrame suitable for
        layered visualizations.

        Args:
            windows: Window size(s) for moving average calculation. Can be a single
                integer or iterable of integers.

        Returns:
            DataFrame in long format with moving averages for each specified window
            size. Series names follow the pattern "ma_X" where X is the window size.

        Examples:
            Multiple moving average window computations:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=50)
            >>> values = np.cumsum(np.random.randn(50))
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> ma_data = tsa._compute_moving_averages([5, 10, 20])
            >>> 'ma_5' in ma_data[tsa._var_name].values
            True
            >>> 'ma_10' in ma_data[tsa._var_name].values
            True
            >>> len(ma_data) == 50 * 3
            True
        """
        results: dict[str, NDArray[np.floating]] = {}
        for i in ensure_iterable(windows):
            series = self.series.rolling(i, min_periods=1, center=True).mean()
            results[f"ma_{i}"] = series

        return self._melt(results)

    def _compute_polynomial_fit(self, degrees: int | Iterable[int]) -> pd.DataFrame:
        """Compute polynomial trend fits of various degrees for trend analysis.

        Fits polynomial trends of different degrees to the time series data using
        least squares regression. Returns a melted DataFrame suitable for layered
        trend visualizations.

        Args:
            degrees: Polynomial degree(s) for trend fitting. Can be a single integer
                or iterable of integers. Higher degrees capture more complex trends
                but may overfit.

        Returns:
            DataFrame in long format with polynomial fits for each specified degree.
            Series names follow the pattern "pf_X" where X is the polynomial degree.

        Examples:
            Multiple polynomial degree trend fitting:

            >>> import pandas as pd
            >>> import numpy as np
            >>> dates = pd.date_range('2025-01-01', periods=100)
            >>> x = np.arange(100)
            >>> values = 0.001 * x**2 + 0.1 * x + np.random.normal(0, 0.5, 100)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date')
            >>> poly_data = tsa._compute_polynomial_fit([1, 2, 3])
            >>> 'pf_1' in poly_data[tsa._var_name].values
            True
            >>> 'pf_2' in poly_data[tsa._var_name].values
            True
            >>> len(poly_data) == 100 * 3
            True
        """
        results: dict[str, NDArray[np.floating]] = {}
        for deg in ensure_iterable(degrees):
            x = np.arange(self.train_size)
            coefs = np.polyfit(x, self.series, deg=deg)
            fit = sum(coef * x**i for i, coef in enumerate(reversed(coefs)))

            results[f"pf_{deg}"] = fit

        return self._melt(results)

    def _melt(self, data: dict[str, NDArray[np.floating] | pd.Series]) -> pd.DataFrame:
        return (
            self.dates.to_frame(self._ds_name)
            .join(pd.DataFrame(data))
            .melt(
                id_vars=self._ds_name,
                var_name=self._var_name,
                value_name=self._value_name,
            )
            .sort_values([self._ds_name, self._var_name], ignore_index=True)
        )

    def _invalidate_cache(self) -> None:
        """Invalidate all cached properties and analysis results after data changes.

        Clears the frequency analysis cache and removes cached properties when
        the underlying data or configuration changes (e.g., when oot_periods
        is modified). This ensures consistency between cached results and the
        current analyzer state.

        Examples:
            Cache invalidation maintains data consistency:

            >>> import pandas as pd
            >>> import numpy as np
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2025-01-01', periods=100),
            ...     'value': np.random.randn(100)
            ... })
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=10)
            >>> initial_train_size = tsa.train_size
            >>> len(tsa._frequency_analysis_cache) == 0
            True
            >>> _ = tsa.find_period_fft()
            >>> len(tsa._frequency_analysis_cache) == 1
            True
            >>> tsa._invalidate_cache()
            >>> len(tsa._frequency_analysis_cache) == 0
            True
        """
        self._frequency_analysis_cache = {}
        cached_props = ["train_data", "oot_data", "dates", "series"]
        for prop in cached_props:
            if prop in self.__dict__:
                del self.__dict__[prop]

    def _repr_mimebundle_(self, include=None, exclude=None) -> MimeBundleType | None:
        """Return a MIME bundle for display in Jupyter notebooks.

        This method is called by Jupyter when the object is displayed.
        It returns an Altair chart visualization of the time series with
        out-of-time periods highlighted if they exist.

        Args:
            include: MIME types to include in the bundle
            exclude: MIME types to exclude from the bundle

        Returns:
            Dictionary with MIME types as keys and content as values, or None

        Examples:
            >>> # In a Jupyter notebook, simply displaying the object will call this method
            >>> import pandas as pd
            >>> import numpy as np
            ...
            >>> dates = pd.date_range('2025-01-01', periods=365, freq='D')
            >>> values = np.sin(2 * np.pi * np.arange(365) / 7) + 0.05 * np.arange(365)
            >>> df = pd.DataFrame({'date': dates, 'value': values})
            >>> tsa = TimeSeriesAnalyzer(df, y='value', ds='date', oot_periods=20)
            ...
            >>> # In a jupyter notebook this will trigger _repr_mimebundle_ to show the visualization
            >>> tsa  # doctest: +SKIP
            alt.LayerChart(...)
        """
        base_chart = alt.Chart(self.df).mark_line().encode(x=self.ds, y=self.y)
        color_chart = None

        if self._oot_periods > 0:
            period_types = np.full(len(self.df), "Training", dtype=object)
            period_types[-self._oot_periods :] = "OOT"
            df = self.df.assign(period_type=period_types).query("period_type=='OOT'")

            base_color = alt.Chart(df).encode(
                x=alt.X(f"{self.ds}:T", title=self._ds_name),
                y=alt.Y(f"{self.y}:Q", title=self._y_name),
                color=alt.Color(
                    "period_type:N",
                    scale=alt.Scale(
                        domain=["Training", "OOT"],
                        range=["#1f77b4", "#d62728"],
                    ),
                    legend=alt.Legend(title="Data Period"),
                ),
            )

            color_chart = (
                base_color.mark_point()
                if self._oot_periods == 1
                else base_color.mark_line()
            )

        return (
            (alt.layer(base_chart, color_chart) if color_chart else base_chart)
            .properties(
                width=self._width,
                height=self._height,
                title=f"Time Series: {self._y_name} over {self._ds_name}",
            )
            ._repr_mimebundle_(include, exclude)
        )
