import warnings
from collections.abc import Callable, Iterable
from dataclasses import InitVar, field
from typing import Annotated, TypeVar

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray
from pydantic import Field, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass

T = TypeVar("T")


@dataclass(slots=True)
class HarmonicFeatureEncoder:
    """Encoder for generating harmonic features from time series or cyclic data.

    Transforms numeric time points into harmonic representations using sine and
    cosine components at different frequencies. This encoding is particularly
    effective for capturing periodic patterns in machine learning models for
    forecasting, regression, and classification tasks.

    The encoder applies harmonic transformations based on a specified period
    length, where each harmonic captures periodic patterns at different scales.
    Lower harmonics represent broader periodic trends, while higher harmonics
    capture finer details and more complex periodic shapes.

    Note on harmonics selection:
        While Fourier series theoretically have infinite harmonics, practical
        implementations must limit the number. The default uses period_length // 2
        harmonics, which corresponds to the Nyquist frequency - the maximum
        frequency that can be uniquely represented given the sampling rate of
        the period. This provides a good balance between capturing periodic
        patterns and avoiding aliasing effects.

    Attributes:
        period_length (int): Number of units in one complete period or cycle.
            Examples: 12 for monthly data with yearly cycles, 7 for daily data
            with weekly cycles, 24 for hourly data with daily cycles, or any
            custom period length. Must be at least 3.
        harmonics (int | None): Number of harmonic frequencies to encode. If None,
            uses period_length // 2 (the Nyquist frequency). More harmonics capture
            finer periodic details but may increase model complexity and potentially
            introduce aliasing if set too high.

    Args:
        precompute (bool): Whether to precompute frequencies and names.
            If False, when calling the methods, the k harmonics must be passed,
            otherwise it will raise ValueError. Default True.

    Examples:
        Basic usage with monthly data:

        >>> encoder = HarmonicFeatureEncoder(period_length=12, harmonics=2)
        >>> encoder.period_length
        12
        >>> encoder.harmonics
        2

        Weekly cycle encoding:

        >>> weekly_encoder = HarmonicFeatureEncoder(period_length=7, harmonics=3)
        >>> features = weekly_encoder.compute_features([1, 2, 3, 4, 5, 6, 7])
        >>> features.shape
        (7, 6)

        Auto-determined harmonics (uses Nyquist frequency):

        >>> auto_encoder = HarmonicFeatureEncoder(period_length=10)
        >>> auto_encoder.harmonics
        5
        >>> auto_encoder.harmonics == auto_encoder.nyquist_frequency
        True
        >>> auto_encoder.base_angular_frequency == 2 * np.pi / 10
        True
    """

    period_length: Annotated[int, Field(ge=3)]
    """Number of units in one complete period or cycle.

    Must be at least greater than 3.
    """
    harmonics: PositiveInt | None = None
    """Number of harmonic frequencies to encode.

    If None, uses the Nyquist frequency (period_length // 2).
    """

    precompute: InitVar[bool] = Field(default=True)
    """Whether to precompute frequencies and names.

    If False, when calling the methods, the k harmonics must be passed. Default True.
    """

    _base_angular_frequency: PositiveFloat = field(init=False, repr=False)
    _nyquist_frequency: PositiveInt = field(init=False, repr=False)
    _frequencies: tuple[float, ...] | None = field(init=False, repr=False)
    _names: tuple[str, ...] | None = field(init=False, repr=False)

    @property
    def base_angular_frequency(self) -> PositiveFloat:
        """The base angular frequency computed as 2π / period_length.

        Returns:
            PositiveFloat: Base angular frequency value.
        """
        return self._base_angular_frequency

    @property
    def nyquist_frequency(self) -> PositiveInt:
        """The Nyquist frequency (period_length // 2).

        This represents the maximum frequency that can be uniquely represented
        without aliasing given the sampling rate defined by the period length.
        It's used as the default number of harmonics when harmonics=None.

        Returns:
            PositiveInt: The Nyquist frequency.
        """
        return self._nyquist_frequency

    def __post_init__(self, precompute: bool) -> None:
        """Initialize computed attributes after dataclass creation.

        Args:
            precompute (bool): Whether to precompute frequencies and names.
        """
        self._nyquist_frequency = self.period_length // 2
        self._base_angular_frequency = 2 * np.pi / self.period_length
        self.harmonics = self.harmonics or self._nyquist_frequency

        if precompute:
            self._frequencies = self._get_frequencies(self.harmonics)
            self._names = self._get_names(self.harmonics)

        else:
            self._frequencies = None
            self._names = None

    def compute_features(
        self,
        t: Iterable[int | float],
        k: PositiveInt | None = None,
        *,
        as_pandas: bool = False,
    ) -> NDArray[np.float64] | pd.DataFrame:
        """Encode time points into harmonic features.

        Transforms time points into harmonic representations using precomputed
        frequencies. The encoding captures periodic patterns through sine and
        cosine components, enabling machine learning models to learn complex
        cyclic behaviors effectively.

        The encoded features are arranged as:
        [sin(h1*t), sin(h2*t), ..., sin(hn*t), cos(h1*t), cos(h2*t), ..., cos(hn*t)]
        where h1, h2, ..., hn are the harmonic frequencies and t are the time points.

        Args:
            t (Iterable[int | float]): Time indices or values to encode. Can be
                any numeric values including non-sequential, fractional, or
                negative numbers.
            k (PositiveInt | None, optional): Number of harmonics to use. If None,
                uses the encoder's harmonics setting. Defaults to None.

        Returns:
            NDArray[np.float64]: 2D array of shape (len(t), 2 * harmonics) with
                encoded harmonic features. First half of columns are sine components,
                second half are cosine components.

        Examples:
            Basic encoding with sequential time points:

            >>> encoder = HarmonicFeatureEncoder(period_length=4, harmonics=2)
            >>> features = encoder.compute_features([1, 2, 3, 4])
            >>> features.shape
            (4, 4)
            >>> features.dtype
            dtype('float64')

            Periodic pattern demonstration:

            >>> encoder = HarmonicFeatureEncoder(period_length=4, harmonics=1)
            >>> cycle_start = encoder.compute_features([1])
            >>> cycle_end = encoder.compute_features([5])  # 1 + period_length
            >>> np.allclose(cycle_start, cycle_end, atol=1e-10)
            True

            Non-sequential time points:

            >>> encoder = HarmonicFeatureEncoder(period_length=4, harmonics=1)
            >>> scattered_features = encoder.compute_features([1, 3, 2, 5])
            >>> scattered_features.shape
            (4, 2)

            Fractional time points:

            >>> encoder = HarmonicFeatureEncoder(period_length=4, harmonics=1)
            >>> fractional_features = encoder.compute_features([1.5, 2.7])
            >>> fractional_features.shape
            (2, 2)
            >>> np.all(np.isfinite(fractional_features))
            np.True_

            Single time point:

            >>> encoder = HarmonicFeatureEncoder(period_length=4, harmonics=1)
            >>> single_feature = encoder.compute_features([1])
            >>> single_feature.shape
            (1, 2)
        """
        frequencies = self._get_or_compute(k, self._frequencies, self._get_frequencies)

        angles = np.outer(np.array(t), frequencies)

        sin_features = np.sin(angles)
        cos_features = np.cos(angles)

        features = np.hstack([sin_features, cos_features])

        if as_pandas:
            return pd.DataFrame(features, index=t, columns=self.get_names(k))

        return features

    def get_names(self, k: PositiveInt | None = None) -> tuple[str, ...]:
        """Get feature names for the encoded harmonic components.

        Args:
            k (PositiveInt | None, optional): Number of harmonics to generate names for.
                If None, uses the encoder's harmonics setting. Defaults to None.

        Returns:
            tuple[str, ...]: Feature names in the format 'sin_h1', 'sin_h2', ...,
                'cos_h1', 'cos_h2', ... corresponding to the columns in the
                output of compute_features().

        Examples:
            Feature names for 2 harmonics:

            >>> encoder = HarmonicFeatureEncoder(period_length=12, harmonics=2)
            >>> names = encoder.get_names()
            >>> names
            ('sin_h1', 'sin_h2', 'cos_h1', 'cos_h2')
            >>> len(names)
            4

            Feature names match feature dimensions:

            >>> encoder = HarmonicFeatureEncoder(period_length=12, harmonics=2)
            >>> features = encoder.compute_features([1, 2, 3])
            >>> names = encoder.get_names()
            >>> len(names) == features.shape[1]
            True
        """
        names = self._get_or_compute(k, self._names, self._get_names)

        return names

    def _get_or_compute(
        self,
        k: PositiveInt | None,
        precomputed_value: T | None,
        compute_fn: Callable[[PositiveInt], T],
    ) -> T:
        """Get precomputed value or compute it using k.

        Args:
            k (PositiveInt | None): Optional parameter to compute value if not precomputed.
            precomputed_value (T | None): Cached value if precompute=True was used.
            compute_fn (Callable[[PositiveInt], T]): Function to compute the value given k.

        Returns:
            T: The precomputed or computed value.

        Raises:
            ValueError: If k is None and no precomputed value exists.
        """
        if k is not None:
            if k > self.nyquist_frequency:
                msg = (
                    "Passing k bigger than the Nyquist Frequency "
                    f"({self.nyquist_frequency}) may result in aliasing."
                )
                warnings.warn(msg, UserWarning)
            return compute_fn(k)

        if precomputed_value is None:
            raise ValueError(
                "k must be provided when precompute=False. "
                "Either set precompute=True during initialization or pass k."
            )

        return precomputed_value

    def _get_frequencies(self, k: PositiveInt) -> tuple[float, ...]:
        """Get harmonic frequencies for k harmonics.

        Args:
            k (PositiveInt): Number of harmonics.

        Returns:
            tuple[float, ...]: Tuple of harmonic frequencies.
        """
        _range = self._get_range(k)
        return tuple(i * self._base_angular_frequency for i in _range)

    def _get_names(self, k: PositiveInt) -> tuple[str, ...]:
        """Get feature names for k harmonics.

        Args:
            k (PositiveInt): Number of harmonics.

        Returns:
            tuple[str, ...]: Tuple of feature names.
        """
        _range = self._get_range(k)
        return tuple(f"{w}_h{i}" for w in ["sin", "cos"] for i in _range)

    @staticmethod
    def _get_range(k: PositiveInt) -> range:
        """Get range for harmonic indices.

        Args:
            k (PositiveInt): Number of harmonics.

        Returns:
            range: Range from 1 to k+1.
        """
        return range(1, k + 1)
