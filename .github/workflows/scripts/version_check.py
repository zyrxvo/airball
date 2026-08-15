"""Version Check for CI."""

import sys
from pathlib import Path

import tomllib  # ty: ignore[unresolved-import]
from packaging.version import Version, parse


def get_version(branch: str) -> Version:
    """Parse the version of the branch from the pyproject.toml."""
    path = Path.cwd() / f"{branch}_branch" / "pyproject.toml"
    return parse(tomllib.loads(path.read_text(encoding="utf-8"))["project"]["version"])


if (curr := get_version("current")) <= (dev := get_version("main")):
    print(f"Error: current version {curr} must be greater than main version {dev}.")
    print("Please bump the version by running 'uv version --bump patch' and pushing the changes.")
    sys.exit(1)
