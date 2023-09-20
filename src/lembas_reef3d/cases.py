import logging
import shutil
import subprocess
from functools import cached_property
from pathlib import Path

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


class Reef3dCase(Case):
    num_processors = InputParameter(default=8)

    @staticmethod
    def log(msg: str, *args, level: int = logging.INFO) -> None:
        logger.log(level, msg, *args)

    @cached_property
    def case_dir(self):
        return Path.cwd().resolve() / "cases" / "case_0"

    @step
    def create_case_dir_if_not_exists(self):
        self.log("Creating case directory: %s", self.case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)

    @step(requires="create_case_dir_if_not_exists")
    def generate_mesh(self):
        self.log("Generating mesh using DIVEMesh")
        mesh_filename = "control.txt"
        shutil.copyfile(TEMPLATE_DIR / mesh_filename, self.case_dir / mesh_filename)

        subprocess.run(["divemesh"], cwd=str(self.case_dir))
