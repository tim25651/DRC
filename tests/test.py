# %%
"""Create test dose and response values."""

from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from drc.drc import ll4

if TYPE_CHECKING:
    from numpy.typing import NDArray


def parse() -> Namespace:
    """Parses CLI arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = ArgumentParser(
        "TestCurve", description="Create test dose and response values"
    )
    parser.add_argument("out", metavar="CSV", help="File to save test data to")
    return parser.parse_args()


def main() -> None:
    """Create test dose and response values."""
    # Set arbitrary parameters
    hill_slope, bottom, top, ec50, duplicates, unit = 1, 1, 3, 1e-6, 5, 1e-6
    base_concs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
    # Set arbitrary concentrations
    concs = np.array(base_concs, dtype=np.float64)
    responses: list[NDArray[np.float64]] = []
    for conc in concs * unit:
        # Calculate responses
        col = duplicates * [ll4(conc, hill_slope, bottom, top, ec50)]
        # Add some random noise
        noise = np.random.normal(1.5, 0.1, size=(duplicates,))
        responses.append(noise * col)

    stacked_responses = np.stack(responses)
    concs_n_responses = np.insert(stacked_responses, 0, concs, axis=1)

    args = parse()

    dr_df = pd.DataFrame(concs_n_responses)
    dr_df.to_csv(args.out, index=False, header=None, float_format="%.4f")


if __name__ == "__main__":
    main()
