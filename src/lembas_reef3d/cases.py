import logging
import subprocess
from functools import cached_property
from pathlib import Path

import toml
from jinja2 import Environment, FileSystemLoader
from lembas import Case, InputParameter, step
from rich.logging import RichHandler

MESH_FILENAME = "control.txt"
CONTROL_FILENAME = "ctrl.txt"

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_level=False, show_path=False, show_time=False)],
)
logger = logging.getLogger(__name__)


TEMPLATE_DIR = Path.cwd().resolve() / "template"
TEMPLATE_ENV = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


class Reef3dCase(Case):
    num_processors = InputParameter(default=8)
    force = InputParameter(default=False)
    wave_height = InputParameter(type=float)
    wave_length = InputParameter(type=float)

    @staticmethod
    def log(msg: str, *args, level: int = logging.INFO) -> None:
        logger.log(level, msg, *args)

    @cached_property
    def case_dir(self):
        return Path.cwd().resolve() / "cases" / f"H={self.wave_height:0.2f}_L={self.wave_length:0.2f}"

    @step(condition=lambda self: not self.case_dir.exists() or self.force)
    def create_case_dir_if_not_exists(self):
        self.log("Creating case directory: %s", self.case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)

    @step(
        requires="create_case_dir_if_not_exists",
        condition=lambda self: not list(self.case_dir.glob("DIVEMesh_*")) or self.force,
    )
    def generate_mesh(self):
        self.log("Generating mesh using DIVEMesh")
        template = TEMPLATE_ENV.get_template(MESH_FILENAME)
        with (self.case_dir / MESH_FILENAME).open("w") as fp:
            fp.write(template.render(num_processors=self.num_processors))

        subprocess.run(["divemesh"], cwd=str(self.case_dir))

    @step(
        requires="generate_mesh",
        condition=lambda self: not list(self.case_dir.glob("REEF3D_*")) or self.force,
    )
    def run_reef3d(self):
        self.log("Running REEF3d")
        template = TEMPLATE_ENV.get_template(CONTROL_FILENAME)
        with (self.case_dir / CONTROL_FILENAME).open("w") as fp:
            fp.write(template.render(case=self))

        subprocess.run(["mpirun", "-n", str(self.num_processors), "reef3d"], cwd=str(self.case_dir))

    def _write_lembas_file(self):
        if self.case_dir:
            self.create_case_dir_if_not_exists()
            data = {"inputs": self.inputs, "case-handler": self.casehandler_full_name}
            with (self.case_dir / "lembas-case.toml").open("w") as fp:
                toml.dump({"lembas": data}, fp)

    def run(self) -> None:
        self._write_lembas_file()  # TODO: move to parent?
        return super().run()
