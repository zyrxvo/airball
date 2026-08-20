"""Version Check for CI."""

import sys
from pathlib import Path

import tomllib  # ty: ignore[unresolved-import]
from packaging.version import Version, parse


def get_version(branch: str) -> Version:
    """Parse the version of the branch from pyproject.toml or __init__.py."""
    root = Path.cwd() / f"{branch}_branch"
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if "version" in pyproject["project"]:
        return parse(pyproject["project"]["version"])

    # TODO: Remove this fallback once main has a static version in pyproject.toml.  # ruff: ignore[line-contains-todo]
    if (init := root / "src" / "airball" / "__init__.py").exists():
        for line in init.read_text(encoding="utf-8").splitlines():
            if "__version__" in line:
                return parse(line.strip().split()[-1].strip("\"'"))

    print(f"Error: could not determine version for {branch} branch.")
    sys.exit(1)


if __name__ == "__main__":
    if (curr := get_version("current")) <= (main := get_version("main")):
        print(f"Error: current version {curr} must be greater than main version {main}.")
        print("Please bump the version by running 'uv version --bump patch' and pushing the changes.")
        sys.exit(1)
