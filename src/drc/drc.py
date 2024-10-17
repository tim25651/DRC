# %%
"""Module to calculate and plot a 4PL-Dose-Response-Curve."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from numpy.typing import NDArray
from scipy.stats.distributions import t
from typing_extensions import Self

from drc.plot import axes, errorbars, labels, plot, scatter, xticks, yticks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

StrPath: TypeAlias = str | os.PathLike
Float32Array: TypeAlias = NDArray[np.float32]

_F = TypeVar("_F", float, Float32Array)


def ll4(x: _F, hill_slope: float, bottom: float, top: float, ec50: float) -> _F:
    """Calculate a reponse value with the fitted 4PL-Dose-Response-Curve.

    This function is basically a copy of the
    LL.4 function from the R drc package with.

    Args:
        x: Dose
        hill_slope: Steepness of curve
        bottom: Lower boundary
        top: Upper boundary
        ec50: Relative EC50

    Returns:
        Either a single response or all responses for the given doses
    """
    response: Float32Array = bottom + (top - bottom) / (
        1 + 10 ** (hill_slope * (np.log10(ec50) - np.log10(x)))
    )
    if isinstance(x, float):
        return float(response)
    return response


@dataclass
class DoseResponseCurve:
    """Storage class for parameter of an 4PL-DoseResponse Curve.

    Y = Bottom + (Top - Bottom) / (1 + 10 ** HillSlope * (lg(EC50) - lg(x)))

    Args:
        hill_slope (float): Steepness of Curve
        bottom (float): Lower boundary
        top (float): Upper boundary
        ec50 (float): Relative EC50 Value
        sds (tuple[float], optional): If provided, SD is stored as well.
        sample_size (int, optional):
            If provided with SD values, CI interval is calculated.
        alpha (float, optional): Level of confidence. Defaults to 0.05.

    Properties:
        log_ec50 (float): Logarithmic EC50 in provided unit
        params (DataFrame): Means, and if initialized SD and CI
    """

    hill_slope: float
    bottom: float
    top: float
    ec50: float
    log_ec50: float = field(init=False)
    sds: tuple[float] | None = field(kw_only=True, default=None, repr=False)
    sample_size: int | None = field(kw_only=True, default=None, repr=False)
    alpha: float = field(kw_only=True, default=0.05, repr=False)

    def __post_init__(self) -> None:
        self.log_ec50 = np.log10(self.ec50)

        self._params = pd.DataFrame(
            {
                "Parameter": ("Hill Slope", "Top", "Bottom", "EC50", "LogEC50"),
                "Mean": (
                    self.hill_slope,
                    self.top,
                    self.bottom,
                    self.ec50,
                    self.log_ec50,
                ),
            }
        )

        if self.sds is not None:
            self._params["SD"] = [*list(self.sds), 0.0]

            if self.sample_size is not None:
                ddof = max(0, self.sample_size - len(self.sds))
                tval = t.ppf(1.0 - self.alpha / 2, ddof)

                ci_lower = list(self.params.Mean[:-1] - tval * self.sds)
                self._params["CI_Lower"] = [*ci_lower, np.log10(ci_lower[-1])]
                ci_upper = list(self.params.Mean[:-1] + tval * self.sds)
                self._params["CI_Upper"] = [*ci_upper, np.log10(ci_upper[-1])]

        self._params.loc[self.params.Parameter == "LogEC50", "SD"] = pd.NA

    @property
    def params(self) -> pd.DataFrame:
        """DataFrame with the parameters of the curve."""
        return self._params


@dataclass
class DoseResponse:
    """Class to calculate a 4PL-Dose-Response-Curve including its parameters.

    Uses a sequence of doses and its corresponding responses
    and plot it with standard error bars.

    Attributes:
        doses (Float32Array): Doses in provided unit
            (if doses are logs, use DoseResponse.from_logs first)
        log_doses (Float32Array): Doses in logarithmic scale in SI
        params (DoseResponseCurve): Parameters of the Dose-Response-Curve
        plot (Figure): Plot of the Dose-Response-Curve
    """

    compound: str
    doses: Float32Array
    responses: Float32Array

    def __post_init__(self) -> None:
        if len(self.doses) != len(self.responses):
            raise ValueError("Different number of doses and responses values")

        self._log_doses = -np.log10(self.doses)

        self._data = pd.DataFrame(
            {"log_dose": self.log_doses, "dose": self.doses, "response": self.responses}
        )

        self._data = self._data.sort_values("dose")

        self._params: DoseResponseCurve | None = None
        self._plot: Figure | None = None

    @staticmethod
    def from_logs(
        log_doses: Float32Array,
        neg: bool = True,
        log_unit: float = 1e-6,
        target_unit: float = 1e-6,
    ) -> Float32Array:
        """Convert (negative) log values in doses in target unit.

        Args:
            log_doses: Log dose values
            neg: If values are the negative log. Defaults to True.
            log_unit: Shift of log values from standard unit. Defaults to 1.0.
            target_unit: Shift of the calculated values. Defaults to 1e-6.

        Returns:
            An array with the doses in provided unit
        """
        if not neg:
            log_doses *= -1

        if log_unit != target_unit:
            log_doses -= np.log10(log_unit) - np.log10(target_unit)

        return 10 ** (-log_doses)

    @property
    def log_doses(self) -> Float32Array:
        """Doses in logarithmic scale in SI."""
        return self._log_doses

    @classmethod
    def read_csv(
        cls,
        filename: StrPath,
        compound: str | None = None,
        dose_col: int | None = None,
        response_cols: Sequence[int] | None = None,
        rm_top_rows: int = 0,
        rm_bottom_rows: int = 0,
    ) -> DoseResponse:
        """Create a DoseResponse Instance from a CSV file.

        Data ending with * are excluded,
        for responses only the single values,
        for doses the complete row

        Args:
            filename: Filename to CSV file,
            compound: Compound name. Defaults to None (Basename of filename used).
            dose_col: Index of columns with doses. Defaults to None (0 used).
            response_cols: Index of columns with responses.
                Defaults to None (every other column but dose column used).
            rm_top_rows: Number of rows to remove from top. Defaults to 0.
            rm_bottom_rows: Number of rows to remove from bottom. Defaults to 0.

        Returns:
            DoseResponse: Dose-Response instance with provided data
        """
        filename = Path(filename)
        if compound is None:
            compound = filename.stem.upper()

        dr_df = pd.read_csv(filename, header=None)

        return cls.read_df(
            dr_df, compound, dose_col, response_cols, rm_top_rows, rm_bottom_rows
        )

    @classmethod
    def read_df(
        cls,
        dr_df: pd.DataFrame,
        compound: str,
        dose_col: int | None = None,
        response_cols: Sequence[int] | None = None,
        rm_top_rows: int = 0,
        rm_bottom_rows: int = 0,
    ) -> Self:
        """Create a DoseResponse Instance from a DataFrame.

        Data ending with * are excluded,
        for responses only the single values,
        for doses the complete row

        Args:
            dr_df: DataFrame with dose and response data as strings
            compound: Compound name. Defaults to None (Basename of filename used).
            dose_col: Index of columns with doses. Defaults to None (0 used).
            response_cols: Index of columsn with responses.
                Defaults to None (every other column but dose column used).
            rm_top_rows: Number of rows to remove from top. Defaults to 0.
            rm_bottom_rows: Number of rows to remove from bottom. Defaults to 0.

        Returns:
            DoseResponse: Dose-Response instance with provided data
        """
        if dose_col is None:
            if response_cols is not None and 0 in response_cols:
                raise ValueError(
                    "Please provide a dolumn for doses, as default = 0 is in responses"
                )

            dose_col = 0

        if response_cols is None:
            response_cols = [col for col in dr_df.columns if col != dose_col]

        dr_df = cls._remove_rows(dr_df, rm_top_rows, rm_bottom_rows)

        doses = dr_df[dose_col]
        doses = cls._exclude_values(doses)
        doses = cls._to_numeric(doses, coerce=False)

        all_doses = []
        all_responses = []
        responses = dr_df[response_cols]
        for col in responses.columns:
            response_col = responses[col]
            response_col = cls._exclude_values(response_col)
            response_col = cls._to_numeric(response_col)

            all_doses.extend(doses)
            all_responses.extend(response_col)

        doses, responses = cls._remove_na(all_doses, all_responses)

        return cls(compound, doses, responses)

    @staticmethod
    def _remove_rows(
        clearable_df: pd.DataFrame, top: int = 0, bottom: int = 0
    ) -> pd.DataFrame:
        """Remove rows from top or/and bottom.

        Args:
            clearable_df: DataFrame to remove rows from
            top: Number of rows to remove from top. Defaults to 0.
            bottom: Number of rows to remove from bottom. Defaults to 0.

        Returns:
            pd.DataFrame: Cropped DataFrame
        """
        bottom = len(clearable_df) if bottom == 0 else -bottom
        return clearable_df[top:bottom]

    @staticmethod
    def _remove_na(doses: Sequence, responses: Sequence) -> tuple[pd.Series, pd.Series]:
        """Remove data points where dose or response is not available or excluded.

        Args:
            doses: Dose values
            responses: Corresponding response values

        Returns:
            A tuple with doses and corresponding responses without NA values
        """
        temp_df = pd.DataFrame({"dose": doses, "response": responses})
        temp_df = temp_df.dropna()
        return temp_df["dose"], temp_df["response"]

    @staticmethod
    def _exclude_values(series: pd.Series, marker: str = "*") -> pd.Series:
        """Exclude values from data if it ends with marker.

        Args:
            series (pd.Series): Series with data points
            marker (str, optional): Marker to mark values to exclude. Defaults to "*".

        Returns:
            pd.Series: Series with np.nan instead of excluded values
        """
        exclude = series.astype(str).str.endswith(marker, na=True)
        return np.where(exclude, np.nan, series)

    @staticmethod
    def _to_numeric(series: pd.Series, coerce: bool = True) -> pd.Series:
        """Convert series to numeric values."""
        try:
            series = pd.to_numeric(series)
        except ValueError as e:
            if coerce:
                print(f"Trying to convert column {series.name} with coerce mode")  # noqa: T201
                series = pd.to_numeric(series, errors="coerce")
            else:
                raise ValueError("Strings in series, coercion turned off") from e

        return series

    def _fit_curve(self) -> None:
        """Fit 4-DL-Dose-Response-Curve with scipy optimizer.curve_fit."""
        self._fit_coefs, self._fit_pcov = opt.curve_fit(
            ll4, self.doses, self.responses, maxfev=100000
        )

    @property
    def params(self) -> DoseResponseCurve:
        """Parameter of the fitted curve."""
        if self._params is None:
            self._params = self.get_params()
        return self._params

    def get_params(self) -> DoseResponseCurve:
        """Get Parameter of the fitted curve.

        Returns:
            The DoseResponseCurve instance with the fitted parameters
        """
        if not hasattr(self, "_fit_coefs"):
            self._fit_curve()

        sds = np.sqrt(np.diag(self._fit_pcov))
        return DoseResponseCurve(*self._fit_coefs, sds=sds, sample_size=len(self.doses))

    @property
    def plot(self) -> Figure:
        """Plot of the fitted curve."""
        if self._plot is None:
            self.get_plot()

        return self._plot

    def get_plot(
        self,
        dose_unit: str = "conc. [µM]",
        response_unit: str = "fold act.",
        title: str | None = None,
        show_vals: bool = False,
        show_errorbars: bool = True,
        adjust_xticks: bool = True,
        adjust_yticks: bool = True,
    ) -> Figure:
        """Plot the fitted curve.

        Args:
            dose_unit: Label of X-axis. Defaults to "conc. [µM]".
            response_unit: Label of Y-axis. Defaults to "fold act.".
            title: Title of plot. Defaults to None (name of the compound is used).
            show_vals: Show all values. Defaults to False.
            show_errorbars: Show standard error bars. Defaults to True.
            adjust_xticks: Adjust ticks to logarithmic scale. Defaults to True.
            adjust_yticks: Adjust yticks to step size of 0.5. Defaults to True.

        Returns:
            Figure: Plot of the fitted curve
        """
        if title is None:
            title = self.compound.upper()

        fig: Figure = plt.Figure()
        ax = fig.add_subplot(111)

        x_fitted, y_fitted = self._get_fitted()

        plot(ax, x_fitted, y_fitted)

        if show_vals:
            scatter(ax, self.log_doses, self.responses)

        if show_errorbars:
            std_x, std_y, std_err = self._get_errors()
            errorbars(ax, std_x, std_y, std_err)

        axes(ax)
        labels(ax, title, dose_unit, response_unit)

        if adjust_xticks:
            xticks(ax, self.log_doses)
        if adjust_yticks:
            yticks(ax, self.responses)

        self._plot = fig
        return self.plot

    def _get_fitted(self) -> tuple[list[float], list[float]]:
        """Get the coordinated of the fitted curve.

        Returns:
            A tuple with two lists, X values and Y values of the fitted curve
        """
        if not hasattr(self, "_fit_coefs"):
            self._fit_curve()

        log_doses_range = np.linspace(min(self.log_doses), max(self.log_doses), 256)
        doses_range = 10**-log_doses_range

        x_fitted = list(log_doses_range)
        y_fitted = [ll4(i, *self._fit_coefs) for i in doses_range]

        return x_fitted, y_fitted

    def _get_errors(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Get coordinates of the error bars.

        Returns:
            A tuple with pd.Series for X values, Y values
            and the size of the standard error bars
        """
        grouped_df = self._data.groupby("log_dose")
        sds = grouped_df.std(ddof=1)
        means = grouped_df.mean()
        sizes = grouped_df.size()

        std_err = sds["response"] / np.sqrt(sizes)

        std_x = pd.Series(means.index)
        std_y = means["response"]

        return std_x, std_y, std_err

    def save_plot(self, save_path: StrPath) -> None:
        """Save the plot of the fitted curve with default settings.

        Args:
            save_path: Path to store the image file to
        """
        self.plot.savefig(save_path)

    def save_params(self, save_path: StrPath) -> None:
        """Save the parameters of the fitted curve as csv file.

        Args:
            save_path: Path to store the csv file to
        """
        self.params.params.to_csv(save_path, index=False, float_format="%.4f")


# %%
