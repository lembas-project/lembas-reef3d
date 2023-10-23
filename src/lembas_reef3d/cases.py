import subprocess
from functools import cache
from functools import cached_property
from pathlib import Path

import pandas as pd
import xarray as xr
from jinja2 import Environment
from jinja2 import FileSystemLoader
from lembas import Case
from lembas import InputParameter
from lembas import step

from lembas_reef3d.postprocessing import load_wave_elevation_line_probe
from lembas_reef3d.postprocessing import load_wave_elevation_time_history

MESH_FILENAME = "control.txt"
CONTROL_FILENAME = "ctrl.txt"

BASE_TEMPLATE_DIR = Path(__file__).parent / "templates"
LOCAL_TEMPLATE_DIR = Path.cwd().resolve() / "templates"
TEMPLATE_ENV = Environment(loader=FileSystemLoader([LOCAL_TEMPLATE_DIR, BASE_TEMPLATE_DIR]))


@cache
def plt():
    import matplotlib.pyplot as plt

    return plt


def result(m):
    return m


class Results:
    wave_time_histories_simulation: pd.DataFrame
    wave_time_histories_theory: pd.DataFrame
    line_probe: xr.DataArray


class RegularWaveCase(Case):
    num_processors = InputParameter(default=8, min=1)
    force = InputParameter(default=False, control=True)
    plot = InputParameter(default=False, control=True)
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
        with pd.HDFStore(self.case_dir / "results.h5") as store:
            try:
                self.results.wave_time_histories_simulation = store.get("wave_time_histories_simulation")
                self.results.wave_time_histories_theory = store.get("wave_time_histories_theory")
            except KeyError:
                self.log("Processing wave elevation results")
                self.results.wave_time_histories_simulation = load_wave_elevation_time_history(
                    self.case_dir / "REEF3D_FNPF_WSF" / "REEF3D-FNPF-WSF-HG.dat"
                )
                self.results.wave_time_histories_theory = load_wave_elevation_time_history(
                    self.case_dir / "REEF3D_FNPF_WSF" / "REEF3D-FNPF-WSF-HG-THEORY.dat"
                )
                store.put("wave_time_histories_simulation", self.results.wave_time_histories_simulation)
                store.put("wave_time_histories_theory", self.results.wave_time_histories_theory)

    @step(requires="run_reef3d")
    def load_line_probe_results(self):
        line_probe_dir = self.case_dir / "REEF3D_FNPF_WSFLINE"

        cdf_path = self.case_dir / "results.cdf"
        if cdf_path.exists():
            arr = xr.open_dataset(cdf_path)["elevation"]
        else:
            arr = xr.concat(
                [load_wave_elevation_line_probe(f) for f in sorted(line_probe_dir.glob("*.dat"))], dim="time"
            )
            arr.attrs["long_name"] = "free-surface elevation"
            arr.attrs["units"] = "m"
            arr.x.attrs["units"] = "m"
            arr.y.attrs["units"] = "m"
            arr.time.attrs["units"] = "s"

            xr.Dataset({"elevation": arr}).to_netcdf(cdf_path)

        self.results.line_probe = arr

    @step(requires="load_wave_results", condition=lambda case: case.plot)
    def plot_wave_results(self):
        ax = plt().gca()
        self.results.wave_time_histories_simulation.plot(ax=ax, style={"P1": "r-", "P2": "b-", "P3": "g-"})
        self.results.wave_time_histories_theory.plot(ax=ax, style={"P1": "r.", "P2": "b.", "P3": "g."})
        plt().show()

    @step(requires="load_line_probe_results", condition=lambda case: case.plot)
    def plot_line_probe_results(self):
        self.results.line_probe.isel(time=-1).plot.line(hue="y")
        plt().show()
