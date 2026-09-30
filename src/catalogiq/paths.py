"""Find checkout-relative paths without relying on a notebook working directory."""
from pathlib import Path
import tomllib


def find_project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            config = candidate / "pyproject.toml"
            if config.is_file():
                with config.open("rb") as stream:
                    project = tomllib.load(stream).get("project", {})
                if project.get("name") == "catalogiq-scope":
                    return candidate
    # An installed wheel can use explicit CLI input/output paths outside a checkout.
    return Path.cwd()


ROOT_PATH = find_project_root()
DATA_PATH = ROOT_PATH / "data"
PROVIDED_DATA_PATH = DATA_PATH / "provided"
TRAIN_PATH = PROVIDED_DATA_PATH / "Selfcare_Training_data (1).csv"
TARGET_PATH = PROVIDED_DATA_PATH / "Selfcare_Target_data (1).csv"
DOCS_PATH = ROOT_PATH / "docs"
NOTEBOOKS_PATH = ROOT_PATH / "notebooks"
