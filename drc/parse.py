from argparse import ArgumentParser, Namespace


def parse() -> Namespace:
    """Parses CLI arguments

    Returns:
        Namespace: Parsed arguments
    """

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
