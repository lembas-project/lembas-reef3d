import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from lembas.logging import logger


def load_wave_elevation_time_history(file: Path) -> pd.DataFrame:
    """Load the wave elevation time histories for a series of wave gauges from a file inside
    the REEF3D_FNPF_WSF directory.

    Returns a dataframe, where the index is the timestamp and each column is one wave probe."""
    with file.open("r") as fp:
        if m := re.match(r"number\s*of\s*gauges:\s*(\d+)", fp.readline()):
            num_gauges = int(m.group(1))
        else:
            num_gauges = 0
        logger.debug(f"num gauges: {num_gauges}")

        fp.readline()  # Blank line
        fp.readline()  # Header: x_coord  y_coord
        coord_data = [fp.readline().split() for _ in range(num_gauges)]

        coord_df = (
            pd.DataFrame(coord_data)
            .rename(columns={0: "point", 1: "x_coord", 2: "y_coord"})
            .astype("float64")
            .astype({"point": "int32"})
            .set_index("point")
        )
        _ = coord_df
        # print(coord_df)

        # Read rest of file as CSV
        df = pd.read_csv(fp, delim_whitespace=True).set_index("time")

        return df


def load_wave_elevation_line_probe(file: Path) -> xr.DataArray:
    """Load the wave elevation results along line probes for a single timestep."""
    with file.open("r") as fp:
        if m := re.match(r"simtime:\s*(\w+)", fp.readline()):
            sim_time = float(m.group(1))
        else:
            sim_time = 0

        logger.debug(f"{sim_time=}")

        if m := re.match(r"number\s*of\s*wsf-lines:\s*(\d+)", fp.readline()):
            num_lines = int(m.group(1))
        else:
            num_lines = 0

        logger.debug(f"{num_lines=}")

        fp.readline()  # Blank line
        fp.readline()  # Header: line_No, y_coord

        # TODO: Load y_coords from file instead of hard-coding
        y_coords = [0.005, np.inf]

        for _ in range(num_lines):
            fp.readline()
        fp.readline()  # Wave Theory

        # Read rest of file as CSV
        fp.readline()
        fp.readline()
        fp.readline()
        df = pd.read_csv(
            fp, delim_whitespace=True, header=None, names=["X", *(f"P{i}" for i in range(1, num_lines + 1)), "W"]
        ).set_index("X")

        return xr.DataArray(df, coords={"x": df.index.to_numpy(), "y": y_coords}, dims=["x", "y"]).expand_dims(
            {"time": [sim_time]}
        )
