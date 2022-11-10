#%%
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .drc import ll4


def parse() -> Namespace:
    """Parses CLI arguments

    Returns:
        Namespace: Parsed arguments
    """

    parser = ArgumentParser(
        "TestCurve", description="Create test dose and response values"
    )
    parser.add_argument("out", metavar="CSV", help="File to save test data to")
    args = parser.parse_args()

    return args


def main():

    # Set arbitrary parameters
    hill_slope, bottom, top, ec50, duplicates, unit = 1, 1, 3, 1e-6, 5, 1e-6
    base_concs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
    # Set arbitrary concentrations
    concs = np.array(base_concs, dtype=np.float32)
    responses: list[NDArray[np.float32]] = []
    for conc in concs * unit:
        # Calculate responses
        col = duplicates * [ll4(conc, hill_slope, bottom, top, ec50)]
        # Add some random noise
        noise = np.random.normal(1.5, 0.1, size=(duplicates,))
        responses.append(noise * col)

    stacked_responses = np.stack(responses)
    concs_n_responses = np.insert(stacked_responses, 0, concs, axis=1)

    args = parse()

    df = pd.DataFrame(concs_n_responses)
    df.to_csv(args.out, index=False, header=None, float_format="%.4f")


if __name__ == "__main__":
    main()
# %%
# %%
