"""Command-line interface for the Dose-Response Curve package."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from drc.drc import DoseResponse


class DoseResponseNamespace(Namespace):
    """Namespace for CLI arguments."""

    file: Path
    out: Path
    dose_col: int | None
    response_cols: list[int] | None


def parse() -> DoseResponseNamespace:
    """Parses CLI arguments.

    Returns:
        Parsed arguments
    """
    parser = ArgumentParser(
        "DRC", description="Fit doses and reponses to a 4PL-Dose-Response-Curve"
    )
    parser.add_argument("file", metavar="CSV", type=Path, help="File to load data from")
    parser.add_argument(
        "-o",
        "--out",
        metavar="DIR",
        type=Path,
        default=".",
        help="Directory to store plot and params to",
    )
    parser.add_argument(
        "-d",
        "--dose_col",
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
    return parser.parse_args()  # type: ignore[return-value]


def main() -> None:
    """Create a Dose-Response curve from a CSV file."""
    args = parse()
    filename = args.file
    dose_col = args.dose_col
    response_cols = args.response_cols
    output_dir = args.out.expanduser().resolve()

    if response_cols is not None:
        response_cols = list(range(response_cols[0], response_cols[1] + 1))

    dr = DoseResponse.read_csv(filename, dose_col=dose_col, response_cols=response_cols)

    lower_compound = dr.compound.lower().replace(" ", "_")
    dr.save_plot(output_dir / f"{lower_compound}_plot.png")
    dr.save_params(output_dir / f"{lower_compound}_params.csv")


if __name__ == "__main__":
    main()
