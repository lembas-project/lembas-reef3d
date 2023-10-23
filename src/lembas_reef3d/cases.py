import subprocess
from functools import cache
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TypeVar

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


TCase = TypeVar("TCase", bound="Case")
RawCaseMethod = Callable[[TCase], None]


@cache
def plt():
    import matplotlib.pyplot as plt

    return plt


def result(*func_or_names: Callable[[TCase], Any] | str):
    """A decorator to annotate a method that provides result(s).

    The decorator accepts a variadic list of names for the provided result(s). The method
    can return a single object or a tuple of objects, which must map to the number of names
    provided. The results are then available from within other case handler methods like
    self.results.result_name.

    """

    if any(callable(fn) for fn in func_or_names):
        # This case captures the non-argument form, i.e. @result
        # In this case, there should only be one argument, which is
        # the decorated method.
        try:
            (method,) = func_or_names
        except ValueError:
            raise ValueError("Must only provide a single callable")
        names = (method.__name__,)  # type: ignore
        method._provides_results = names  # type: ignore
        return method

    # We now handle the case with arguments, i.e. @result("name1", "name2")
    names = func_or_names  # type: ignore

    def decorator(m: RawCaseMethod):
        # Here, we attach the tuple of names to the method function object. We have to do
        # this because we do not have access to the class object at the time when the
        # decorator is called. The actual discovery of the name mapping is performed
        # during attribute access on the case.results object, which does a search across
        # the methods attached to the class at runtime.
        m._provides_results = names  # type: ignore
        return m

    return decorator


class Results:
    """A generic container for results of a case.

    Implements lazy loading of results, where the result accessors are specified by @result
    decorator.

    """

    def __init__(self, parent: Case):
        self._parent = parent

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__:
            return self.__dict__[item]

        # During attribute access, we search the class for methods to which have been
        # attached a "_provides_results" tuple. If we find that, and the requested
        # result is in that tuple, we call the method (once) and cache the results in
        # the self.__dict__ for later, faster retrieval.
        cls = self._parent.__class__
        for method_name, method_func in cls.__dict__.items():
            try:
                provides_results = getattr(method_func, "_provides_results")
            except AttributeError:
                continue  # to next method

            provides_results = provides_results or tuple()
            if item not in provides_results:
                continue

            results = method_func(self._parent)
            if not isinstance(results, tuple):
                results = (results,)

            num_expected_results = len(provides_results)
            num_results = len(results)
            if num_expected_results != num_results:
                raise ValueError(
                    f"Results method {method_name} returns {num_results} items, "
                    f"only {num_expected_results} results are declared in the @result "
                    "decorator."
                )

            for n, r in zip(provides_results, results):
                setattr(self, n, r)

        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError(f"Result '{item}' is not defined")


class RegularWaveCase(Case):
    num_processors = InputParameter(default=8, min=1)
    force = InputParameter(default=False, control=True)
    plot = InputParameter(default=False, control=True)
    skip_plot = InputParameter(default=False, control=True)
    wave_height = InputParameter(type=float, min=0.0)
    wave_length = InputParameter(type=float, min=0.0)

    def __init__(self, **kwargs: Any):
        # TODO: Move this and Results class to lembas-core
        super().__init__(**kwargs)
        self.results = Results(parent=self)

    @cached_property
    def case_dir(self) -> Path:
        return (
            Path.cwd().resolve()
            / "cases"
            / f"H={self.wave_height:0.2f}_L={self.wave_length:0.2f}_np={self.num_processors}"
        )

    @cached_property
    def fig_dir(self) -> Path:
        return self.case_dir / "figures"

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

    @result("wave_time_histories_simulation", "wave_time_histories_theory")
    def load_wave_results(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with pd.HDFStore(self.case_dir / "results.h5") as store:
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
        cdf_path = self.case_dir / "results.cdf"
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
