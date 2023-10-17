import re
import subprocess
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import xarray
import xarray as xr
from jinja2 import Environment
from jinja2 import FileSystemLoader
from lembas import Case
from lembas import InputParameter
from lembas import step
from lembas.logging import logger
from matplotlib import pyplot

MESH_FILENAME = "control.txt"
CONTROL_FILENAME = "ctrl.txt"

BASE_TEMPLATE_DIR = Path(__file__).parent / "templates"
LOCAL_TEMPLATE_DIR = Path.cwd().resolve() / "templates"
TEMPLATE_ENV = Environment(loader=FileSystemLoader([LOCAL_TEMPLATE_DIR, BASE_TEMPLATE_DIR]))


def result(m):
    return m


class Results:
    wave_time_histories_simulation: pandas.DataFrame
    wave_time_histories_theory: pandas.DataFrame
    line_probe: xarray.DataArray


class RegularWaveCase(Case):
    num_processors = InputParameter(default=8, min=1)
    force = InputParameter(default=False, control=True)
    wave_height = InputParameter(type=float, min=0.0)
    wave_length = InputParameter(type=float, min=0.0)

    @cached_property
    def case_dir(self) -> Path:
        return (
            Path.cwd().resolve()
            / "cases"
            / f"H={self.wave_height:0.2f}_L={self.wave_length:0.2f}_np={self.num_processors}"
        )

    def has_case_files(self, glob_pattern: str) -> bool:
        """Return True if there is at least one file matching the provided glob pattern inside the case directory."""
        return bool(list(self.case_dir.glob(glob_pattern)))

    @step(condition=lambda self: not self.case_dir.exists() or self.force)
    def create_case_dir_if_not_exists(self):
        self.log("Creating case directory: %s", self.case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)

    @step(
        requires="create_case_dir_if_not_exists",
        condition=lambda self: not self.has_case_files("DIVEMesh_*") or self.force,
    )
    def generate_mesh(self):
        self.log("Generating mesh using DIVEMesh")
        template = TEMPLATE_ENV.get_template(MESH_FILENAME)
        with (self.case_dir / MESH_FILENAME).open("w") as fp:
            fp.write(template.render(num_processors=self.num_processors))

        subprocess.run(["divemesh"], cwd=str(self.case_dir))

    @step(
        requires="generate_mesh",
        condition=lambda self: not self.has_case_files("REEF3D_*") or self.force,
    )
    def run_reef3d(self):
        self.log("Running REEF3d")
        template = TEMPLATE_ENV.get_template(CONTROL_FILENAME)
        with (self.case_dir / CONTROL_FILENAME).open("w") as fp:
            fp.write(template.render(case=self))

        subprocess.run(["mpirun", "-n", str(self.num_processors), "reef3d"], cwd=str(self.case_dir))

    @step(requires="run_reef3d")
    @result
    def load_wave_results(self):
        self.results = Results()
        with pandas.HDFStore(self.case_dir / "results.h5") as store:
            try:
                self.results.wave_time_histories_simulation = store.get("wave_time_histories_simulation")
                self.results.wave_time_histories_theory = store.get("wave_time_histories_theory")
            except KeyError:
                self.log("Processing wave elevation results")
                self.results.wave_time_histories_simulation = _load_wave_elevation_time_history(
                    self.case_dir / "REEF3D_FNPF_WSF" / "REEF3D-FNPF-WSF-HG.dat"
                )
                self.results.wave_time_histories_theory = _load_wave_elevation_time_history(
                    self.case_dir / "REEF3D_FNPF_WSF" / "REEF3D-FNPF-WSF-HG-THEORY.dat"
                )
                store.put("wave_time_histories_simulation", self.results.wave_time_histories_simulation)
                store.put("wave_time_histories_theory", self.results.wave_time_histories_theory)

    @step(requires="run_reef3d")
    def load_line_probe_results(self):
        line_probe_dir = self.case_dir / "REEF3D_FNPF_WSFLINE"

        cdf_path = self.case_dir / "results.cdf"
        if cdf_path.exists():
            full_array = xr.open_dataset(cdf_path)["elevation"]
        else:
            time_slices = []
            for f in sorted(line_probe_dir.glob("*.dat")):
                sim_time, y_coords, df = _load_wave_elevation_line_probe(f)
                arr = xr.DataArray(df, coords={"x": df.index.to_numpy(), "y": y_coords}, dims=["x", "y"]).expand_dims(
                    {"time": [sim_time]}
                )
                time_slices.append(arr)

            full_array = xr.concat(time_slices, dim="time")
            full_array.attrs["long_name"] = "free-surface elevation"
            full_array.attrs["units"] = "m"
            full_array.x.attrs["units"] = "m"
            full_array.y.attrs["units"] = "m"
            full_array.time.attrs["units"] = "s"

            xr.Dataset({"elevation": full_array}).to_netcdf(cdf_path)

        self.results.line_probe = full_array

    @step(requires="load_wave_results")
    def plot_wave_results(self):
        ax = pyplot.gca()
        self.results.wave_time_histories_simulation.plot(ax=ax, style={"P1": "r-", "P2": "b-", "P3": "g-"})
        self.results.wave_time_histories_theory.plot(ax=ax, style={"P1": "r.", "P2": "b.", "P3": "g."})

        pyplot.show()

    @step(requires="load_line_probe_results")
    def plot_line_probe_results(self):
        self.results.line_probe.isel(time=-1).plot.line(hue="y")
        plt.show()


def _load_wave_elevation_time_history(file: Path) -> pandas.DataFrame:
    """Load the wave elevation time histories for a series of wave gauges from a file inside
    the REEF3D_FNPF_WSF directory.

    Returns a dataframe, where the index is the timestamp and each column is one wave probe."""
    with file.open("r") as fp:
        if m := re.match(r"number\s*of\s*gauges:\s*(\d+)", fp.readline()):
            num_gauges = int(m.group(1))
        else:
            num_gauges = 0
        logger.info(f"num gauges: {num_gauges}")

        fp.readline()  # Blank line
        fp.readline()  # Header: x_coord  y_coord
        coord_data = [fp.readline().split() for _ in range(num_gauges)]

        coord_df = (
            pandas.DataFrame(coord_data)
            .rename(columns={0: "point", 1: "x_coord", 2: "y_coord"})
            .astype("float64")
            .astype({"point": "int32"})
            .set_index("point")
        )
        _ = coord_df
        # print(coord_df)

        # Read rest of file as CSV
        df = pandas.read_csv(fp, delim_whitespace=True).set_index("time")

        return df


def _load_wave_elevation_line_probe(file: Path) -> pandas.DataFrame:
    print(file)
    with file.open("r") as fp:
        if m := re.match(r"simtime:\s*(\w+)", fp.readline()):
            sim_time = float(m.group(1))
        else:
            sim_time = 0

        logger.info(f"{sim_time=}")

        if m := re.match(r"number\s*of\s*wsf-lines:\s*(\d+)", fp.readline()):
            num_lines = int(m.group(1))
        else:
            num_lines = 0

        logger.info(f"{num_lines=}")

        fp.readline()  # Blank line
        fp.readline()  # Header: line_No, y_coord

        for _ in range(num_lines):
            fp.readline()
        fp.readline()  # Wave Theory

        # Read rest of file as CSV
        fp.readline()
        fp.readline()
        fp.readline()
        df = (
            pandas.read_csv(
                fp, delim_whitespace=True, header=None, names=["X", *(f"P{i}" for i in range(1, num_lines + 1)), "W"]
            )
            # .drop(columns=["W"])
            .set_index("X")
        )
        print(df)

        return sim_time, [0.005, np.inf], df
