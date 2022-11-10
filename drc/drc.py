#%%
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats.distributions import t


def ll4(x, hill_slope, bottom, top, ec50):
    """This function is basically a copy of the LL.4 function from the R drc package with
    - hill_slope: hill slope
    - bottom: min response
    - top: max response
    - ec50: EC50"""
    return bottom + (top - bottom) / (
        1 + 10 ** (hill_slope * (np.log10(ec50) - np.log10(x)))
    )


@dataclass
class DoseResponseCurve:
    """Storage class for parameter of an 4PL-DoseResponse Curve

    Y = Bottom + (Top - Bottom) / (1 + 10 ** HillSlope * (lg(EC50) - lg(x)))

    Args:
        hill_slope (float): Steepness of Curve
        bottom (float): Lower boundary
        top (float): Upper boundary
        ec50 (float): Relative EC50 Value
        unit (float, optional): Scaling coefficient. Defaults to 1e-6.
        sds (tuple[float], optional): If provided, SD is stored as well.
        sample_size (int, optional): If provided with SD values, CI interval is calculated.
        in_unit (bool, optional): If values are in provided unit. Defaults to True.
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
    unit: float = 1e-6
    sds: tuple[float] = field(kw_only=True, default=None, repr=False)
    sample_size: int = field(kw_only=True, default=None, repr=False)
    in_unit: bool = field(kw_only=True, default=False, repr=False)
    alpha: float = field(kw_only=True, default=0.05, repr=False)

    def __post_init__(self):

        if not self.in_unit:
            self.ec50 = self.ec50 / self.unit
            self._log_ec50 = np.log10(self.ec50)
        else:
            self._log_ec50 = np.log10(self.ec50 * self.unit)

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
            if not self.in_unit:
                self.sds[-1] = self.sds[-1] / self.unit
            self._params["SD"] = list(self.sds) + [0.0]

            if self.sample_size is not None:
                ddof = max(0, self.sample_size - len(self.sds))
                tval = t.ppf(1.0 - self.alpha / 2, ddof)

                ci_lower = list(self.params.Mean[:-1] - tval * self.sds)
                self._params["CI_Lower"] = ci_lower + [np.log10(ci_lower[-1])]
                ci_upper = list(self.params.Mean[:-1] + tval * self.sds)
                self._params["CI_Upper"] = ci_upper + [np.log10(ci_upper[-1])]

        self._params.loc[self.params.Parameter == "LogEC50", "SD"] = pd.NA

    @property
    def params(self):
        return self._params

    @property
    def log_ec50(self):
        return self._log_ec50

    @log_ec50.setter
    def log_ec50(self, value):
        self._log_ec50 = value


@dataclass
class DoseResponse:
    """Class to calculate a 4PL-Dose-Response-Curve including its parameters
    from a sequences of doses and its corresponding responses
    and plot it with standard error bars.

    Args:
        compound (str): Name of compound
        doses (Sequence[float]): Doses in provided unit
        responses (Sequence[float]): Responses
        unit (float, optional): Scaling coefficient. Defaults to 1e-6.
        log (int, optional): 0 (not logarithmic), 1 (neg. logs in SI), 2 (logs in SI), 3 (neg. logs in unit), 4 (logs in unit), Defaults to 0.

    Properties:
        doses (NDArray): Doses in SI
        log_doses (NDArray): Doses in logarithmic scale in SI
        params (DoseResponseCurve): Parameters of the Dose-Response-Curve
        plot (Figure): Plot of the Dose-Response-Curve
    """

    compound: str
    doses: Sequence[float]
    responses: Sequence[float]
    unit: float = 1e-6
    log: int = field(kw_only=True, repr=False, default=0)

    def __post_init__(self):
        assert len(self.doses) == len(
            self.responses
        ), "Different number of doses and responses values"

        self.doses = np.array(self.doses, dtype=np.float32) * self.unit

        if self.log not in (0, 1, 2, 3, 4):
            raise ValueError("Argument logarithmic has to be either 0, 1, 2, 3 or 4.")
        elif self.log == 0:
            self._log_doses = -np.log10(self.doses)
        else:
            self._log_doses = self.doses
            if self.log in (2, 4):
                self._log_doses *= -1
            if self.log in (1, 3):
                scaling = -np.log10(self.unit)
                self._log_doses += scaling
            self.doses = 10 ** (-self.log_doses)

        self.responses = np.array(self.responses, dtype=np.float32)

        sorted_indices = self.doses.argsort()
        self.doses.sort()
        self._log_doses = self.log_doses[sorted_indices]
        self.responses = self.responses[sorted_indices]

        self._data = pd.DataFrame(
            {"log_dose": self.log_doses, "dose": self.doses, "response": self.responses}
        )

        self._params = None
        self._plot = None

    @property
    def log_doses(self):
        return self._log_doses

    @classmethod
    def read_csv(
        cls,
        filename: str,
        compound: str = None,
        dose_col: int = None,
        response_cols: Iterable[int] = None,
        rm_top_rows: int = 0,
        rm_bottom_rows: int = 0,
    ) -> "DoseResponse":
        """Create a DoseResponse Instance from a CSV file

        Data ending with * are excluded, for responses only the single values, for doses the complete row

        Args:
            filename (str): Filename to CSV file,
            compound (str, optional): Compound name. Defaults to None (Basename of filename used).
            dose_col (int, optional): Index of columns with doses. Defaults to None (0 used).
            response_cols (Iterable[int], optional): Index of columsn with responses. Defaults to None (every other column but dose column used).
            rm_top_rows (int, optional): Number of rows to remove from top. Defaults to 0.
            rm_bottom_rows (int, optional): Number of rows to remove from bottom. Defaults to 0.

        Returns:
            DoseResponse: Dose-Response instance with provided data
        """
        if compound is None:
            basename = os.path.basename(filename)
            basename_wo_ext = "".join(basename.split(".")[:-1])
            compound = basename_wo_ext.upper()

        df = pd.read_csv(filename, header=None)

        return cls.read_df(
            df, compound, dose_col, response_cols, rm_top_rows, rm_bottom_rows
        )

    @classmethod
    def read_df(
        cls,
        df: pd.DataFrame,
        compound: str,
        dose_col: int = None,
        response_cols: Iterable[int] = None,
        rm_top_rows: int = 0,
        rm_bottom_rows: int = 0,
    ) -> "DoseResponse":
        """Create a DoseResponse Instance from a DataFrame

        Data ending with * are excluded, for responses only the single values, for doses the complete row

        Args:
            df (pd.DataFrame): DataFrame with dose and response data as strings
            compound (str, optional): Compound name. Defaults to None (Basename of filename used).
            dose_col (int, optional): Index of columns with doses. Defaults to None (0 used).
            response_cols (Iterable[int], optional): Index of columsn with responses. Defaults to None (every other column but dose column used).
            rm_top_rows (int, optional): Number of rows to remove from top. Defaults to 0.
            rm_bottom_rows (int, optional): Number of rows to remove from bottom. Defaults to 0.

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
            response_cols = df.columns[1:]

        if rm_bottom_rows == 0:
            rm_bottom_rows = len(df)
        else:
            rm_bottom_rows = -rm_bottom_rows
        df = df[rm_top_rows:rm_bottom_rows]

        doses = df[dose_col]
        doses = cls._exclude_values(doses)
        doses = cls._to_numeric(doses, coerce=False)
        _all_doses = []
        _all_responses = []
        responses: pd.DataFrame = df[response_cols]
        for col in responses.columns:
            response_col = responses[col]
            response_col = cls._exclude_values(response_col)
            response_col = cls._to_numeric(response_col)

            _all_doses.extend(doses)
            _all_responses.extend(response_col)

        all_doses = np.array(_all_doses)
        all_responses = np.array(_all_responses)
        not_nan = ~np.isnan(all_responses)
        all_doses = all_doses[not_nan]
        all_responses = all_responses[not_nan]

        return cls(compound, all_doses, all_responses)

    @staticmethod
    def _exclude_values(series: pd.Series, marker: str = "*"):
        exclude = series.astype(str).str.endswith(marker, na=True)
        series = np.where(exclude, np.nan, series)
        return series

    @staticmethod
    def _to_numeric(series: pd.Series, coerce=True):
        try:
            series = pd.to_numeric(series)
        except ValueError as v:
            if coerce:
                print(f"Trying to convert column {series.name} with coerce mode")
                series = pd.to_numeric(series, errors="coerce")
            else:
                raise v("Strings in series, coercion turned off")

        return series

    def _fit_curve(self):
        self._fit_coefs, self._fit_pcov = opt.curve_fit(
            ll4,
            self.doses,
            self.responses,
            maxfev=100000,  # p0=p0
        )

    @property
    def params(self):
        if self._params is None:
            self.get_params()
        return self._params

    def get_params(self):
        if not hasattr(self, "_fit_coefs"):
            self._fit_curve()

        sds = np.sqrt(np.diag(self._fit_pcov))
        self._params = DoseResponseCurve(
            *self._fit_coefs, sds=sds, sample_size=len(self.doses)
        )

        return self.params.params

    @property
    def plot(self):
        if self._plot is None:
            self.get_plot()

        return self._plot

    def get_plot(
        self,
        dose_unit: str = "conc. [µM]",
        response_unit: str = "fold act.",
        title: str | None = None,
    ):
        if title is None:
            title = self.compound.upper()
        if not hasattr(self, "_fit_coefs"):
            self._fit_curve()

        log_doses_range = np.linspace(min(self.log_doses), max(self.log_doses), 256)
        doses_range = 10**-log_doses_range

        x_fitted = log_doses_range
        y_fitted = [ll4(i, *self._fit_coefs) for i in doses_range]
        plt.plot(x_fitted, y_fitted, color="black")

        if False:
            self._set_values()

        if True:
            self._set_errorbars()

        ax = plt.gca()
        self._set_axis(ax)

        plt.title(title, weight="bold", size="18")
        plt.xlabel(dose_unit, weight="bold", size="16")
        plt.ylabel(response_unit, weight="bold", size="16")

        self._set_xticks()

        if True:
            self._set_yticks()

        self._plot = plt.gcf()

        return self.plot

    def _set_values(self):
        plt.scatter(self.log_doses, self.responses)

    def _set_errorbars(self):
        grouped_df = self._data.groupby("log_dose")
        stds = grouped_df.std(ddof=1)
        sizes = grouped_df.size()

        std_err = stds.response / np.sqrt(sizes)

        std_x = stds.index
        std_y = [ll4(i, *self._fit_coefs) for i in 10 ** (-std_x)]
        plt.errorbar(
            std_x,
            std_y,
            std_err,
            color="black",
            marker="o",
            capsize=4,
            linewidth=0,
            elinewidth=2,
        )

    def _set_axis(self, ax):
        ax.invert_xaxis()
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines["left"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.tick_params(width=2)

    def _set_yticks(self):
        max_val = np.ceil(max(self.responses) * 2).astype(int)
        min_val = np.floor(min(self.responses) * 2).astype(int)
        y_ticks = [x / 2 for x in range(min_val, max_val + 1)][1:-1]
        plt.yticks(y_ticks)

    def _set_xticks(self):
        min_val = 10 ** np.floor(-max(self.log_doses))
        max_val = 10 ** np.ceil(-min(self.log_doses))

        x_ticks = [f * min_val for f in (0.7, 0.8, 0.9, 1.0)]
        x_tick_labels = ["", "", "", f"{min_val/self.unit:.2f}"]
        while min_val <= max_val:
            for f in range(2, 10):
                x_ticks.append(f * min_val)
                x_tick_labels.append("")
            min_val = 10 * min_val
            x_ticks.append(min_val)
            label = (
                f"{min_val/self.unit:.2f}"
                if min_val < self.unit
                else f"{min_val/self.unit:.0f}"
            )
            x_tick_labels.append(label)
            if (max_val - min_val) < 1e-15:
                break
        x_ticks = -np.log10(x_ticks)
        plt.xticks(x_ticks, x_tick_labels, weight="bold")
        plt.yticks(weight="bold")

    def save_plot(self, save_path: str):
        self.plot.savefig(save_path)

    def save_params(self, save_path: str):
        self.params.params.to_csv(save_path, index=False, float_format="%.4f")


def parse():
    parser = ArgumentParser(
        "DRC", description="Fit doses and reponses to a 4PL-Dose-Response-Curve"
    )
    parser.add_argument("file", metavar="CSV", help="File to load data from")
    parser.add_argument(
        "-o",
        "--out",
        metavar="DIR",
        default=".",
        help="Directory to store plot and params to",
    )
    parser.add_argument(
        "-d",
        "--dose_col",
        nargs=1,
        metavar="COL",
        type=int,
        default=None,
        help="Column index of doses",
    )
    parser.add_argument(
        "-r",
        "--response_cols",
        metavar="COL",
        type=int,
        nargs=2,
        default=None,
        help="Column index of range begin and end of responses",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse()
    filename = args.file
    dose_col: list[int] | None = args.dose_col
    response_cols: list[int] | None = args.response_cols
    output_dir = args.out

    if dose_col is not None:
        dose_col = dose_col[0]
    if response_cols is not None:
        response_cols = list(range(response_cols[0], response_cols[1] + 1))

    dr = DoseResponse.read_csv(filename, dose_col=dose_col, response_cols=response_cols)

    output_base = output_dir + "/" + dr.compound
    dr.save_plot(output_base.lower() + "_plot.png")
    dr.save_params(output_base.lower() + "_params.csv")


if __name__ == "__main__":
    main()
# %%
