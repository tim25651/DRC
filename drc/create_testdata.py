#%%
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from .drc import ll4


def main():
    hill_slope, bottom, top, ec50 = 1, 1, 3, 1e-6
    concs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
    cols = {}
    for conc in concs:
        resp = ll4(conc * 1e-6, hill_slope, bottom, top, ec50)
        for n in range(1, 5):
            if n not in cols:
                cols[n]: list[float] = []
            x = np.random.normal(1.0, 0.1)
            cols[n].append(x * resp)

    parser = ArgumentParser(
        "TestCurve", description="Create test dose and response values"
    )
    parser.add_argument("out", metavar="CSV", help="File to save test data to")
    args = parser.parse_args()

    df = pd.DataFrame({0: concs, **cols})
    df.to_csv(args.out, index=False, header=None, float_format="%.4f")


if __name__ == "__main__":
    main()
# %%
# %%
