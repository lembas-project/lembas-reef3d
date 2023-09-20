import logging
import subprocess
from functools import cached_property
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from lembas import Case, InputParameter, step
from rich.logging import RichHandler

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

    @staticmethod
    def log(msg: str, *args, level: int = logging.INFO) -> None:
        logger.log(level, msg, *args)

    @cached_property
    def case_dir(self):
        return Path.cwd().resolve() / "cases" / "case_0"

    @step(condition=lambda self: not self.case_dir.exists())
    def create_case_dir_if_not_exists(self):
        self.log("Creating case directory: %s", self.case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)

    @step(
        requires="create_case_dir_if_not_exists",
        condition=lambda self: not list(self.case_dir.glob("DIVEMesh_*")),
    )
    def generate_mesh(self):
        self.log("Generating mesh using DIVEMesh")
        mesh_filename = "control.txt"
        template = TEMPLATE_ENV.get_template(mesh_filename)
        with (self.case_dir / mesh_filename).open("w") as fp:
            fp.write(template.render(num_processors=self.num_processors))

        subprocess.run(["divemesh"], cwd=str(self.case_dir))

    @step(
        requires="generate_mesh",
        condition=lambda self: not list(self.case_dir.glob("REEF3D_*")),
    )
    def run_reef3d(self):
        self.log("Running REEF3d")
        mesh_filename = "ctrl.txt"
        template = TEMPLATE_ENV.get_template(mesh_filename)
        with (self.case_dir / mesh_filename).open("w") as fp:
            fp.write(template.render(num_processors=self.num_processors))

        subprocess.run(["mpirun", "-n", str(self.num_processors), "reef3d"], cwd=str(self.case_dir))
