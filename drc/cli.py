from . import DoseResponse
from .parse import parse


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
