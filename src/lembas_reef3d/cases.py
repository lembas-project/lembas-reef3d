import subprocess
import sys
from functools import cache
from functools import cached_property
from pathlib import Path

import pandas as pd
import xarray as xr
from jinja2 import Environment
from jinja2 import FileSystemLoader
from lembas import Case
from lembas import InputParameter
from lembas import result
from lembas import step

from lembas_reef3d.postprocessing import load_wave_elevation_line_probe
from lembas_reef3d.postprocessing import load_wave_elevation_time_history

MESH_FILENAME = "control.txt"
CONTROL_FILENAME = "ctrl.txt"

BASE_TEMPLATE_DIR = Path(__file__).parent / "templates"
LOCAL_TEMPLATE_DIR = Path.cwd().resolve() / "templates"
TEMPLATE_ENV = Environment(loader=FileSystemLoader([LOCAL_TEMPLATE_DIR, BASE_TEMPLATE_DIR]))

# divemesh and reef3d should be in same directory as python in a conda environment
BIN_DIR = Path(sys.executable).parent


@cache
def plt():
    import matplotlib.pyplot as plt

    return plt


class RegularWaveCase(Case):
    num_processors = InputParameter(default=8, min=1)
    force = InputParameter(default=False, control=True)
    plot = InputParameter(default=False, control=True)
    skip_plot = InputParameter(default=False, control=True)
    wave_height = InputParameter(type=float, min=0.0)
    wave_length = InputParameter(type=float, min=0.0)

    @cached_property
    def case_dir(self) -> Path:
        return (
            Path.cwd().resolve()
            / "cases"
            / f"H={self.wave_height:0.2f}_L={self.wave_length:0.2f}_np={self.num_processors}"
        )

    @cached_property
    def data_dir(self) -> Path:
        """The base directory in which to store results generated by lembas."""
        return self.case_dir / "lembas" / "data"

    @cached_property
    def fig_dir(self) -> Path:
        """The base directory in which to save post-processing figures."""
        return self.case_dir / "lembas" / "figures"

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

        subprocess.run([str(BIN_DIR / "divemesh")], cwd=str(self.case_dir))

    @step(
        requires="generate_mesh",
        condition=lambda self: not self.has_case_files("REEF3D_*") or self.force,
    )
    def run_reef3d(self):
        self.log("Running REEF3d")
        template = TEMPLATE_ENV.get_template(CONTROL_FILENAME)
        with (self.case_dir / CONTROL_FILENAME).open("w") as fp:
            fp.write(template.render(case=self))

        subprocess.run([str(BIN_DIR / "mpirun"), "-n", str(self.num_processors), "reef3d"], cwd=str(self.case_dir))

    @result("wave_time_histories_simulation", "wave_time_histories_theory")
    def load_wave_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        store_path = self.data_dir / "time_histories.h5"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        with pd.HDFStore(store_path) as store:
            try:
                wave_time_histories_simulation = store.get("wave_time_histories_simulation")
                wave_time_histories_theory = store.get("wave_time_histories_theory")
            except KeyError:
                self.log("Processing wave elevation results")
                wave_time_histories_simulation = load_wave_elevation_time_history(
                    self.case_dir / "REEF3D_FNPF_WSF" / "REEF3D-FNPF-WSF-HG.dat"
                )
                wave_time_histories_theory = load_wave_elevation_time_history(
                    self.case_dir / "REEF3D_FNPF_WSF" / "REEF3D-FNPF-WSF-HG-THEORY.dat"
                )
                store.put("wave_time_histories_simulation", wave_time_histories_simulation)
                store.put("wave_time_histories_theory", wave_time_histories_theory)
            return wave_time_histories_simulation, wave_time_histories_theory

    @result
    def line_probes(self) -> xr.DataArray:
        cdf_path = self.data_dir / "probe_results.cdf"
        if cdf_path.exists():
            arr = xr.open_dataset(cdf_path)["elevation"]
        else:
            line_probe_dir = self.case_dir / "REEF3D_FNPF_WSFLINE"
            arr = xr.concat(
                [load_wave_elevation_line_probe(f) for f in sorted(line_probe_dir.glob("*.dat"))], dim="time"
            )
            arr.attrs["long_name"] = "free-surface elevation"
            arr.attrs["units"] = "m"
            arr.x.attrs["units"] = "m"
            arr.y.attrs["units"] = "m"
            arr.time.attrs["units"] = "s"

            cdf_path.parent.mkdir(exist_ok=True, parents=True)
            xr.Dataset({"elevation": arr}).to_netcdf(cdf_path)

        return arr

    @step(requires="run_reef3d", condition="plot")
    def plot_wave_results(self):
        fig_path = self.fig_dir / "time_history.png"
        if not fig_path.exists():
            fig, ax = plt().subplots(1, 1)
            self.results.wave_time_histories_theory.plot(ax=ax, style={"P1": "r-", "P2": "b-", "P3": "g-"})
            self.results.wave_time_histories_simulation.plot(ax=ax, style={"P1": "r.", "P2": "b.", "P3": "g."})

            self.fig_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)

    @step(requires="run_reef3d", condition="not skip_plot")
    def plot_line_probe_results(self):
        fig_path = self.fig_dir / "probe_results.png"
        if not fig_path.exists():
            fig, ax = plt().subplots(1, 1)
            self.results.line_probes.isel(time=-1).plot.line(ax=ax, hue="y")

            self.fig_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(fig_path)
