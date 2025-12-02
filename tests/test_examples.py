"""Test the Documentation examples to make sure they work correctly."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parents[1] / "docs" / "examples"
EXAMPLES = sorted((path for path in ROOT_DIR.glob("*.ipynb")), reverse=True)


@pytest.fixture
def convert_ipynb_to_py(request, tmp_path: Path) -> Path:
    """Convert an ipynb to a python script."""
    filename = Path(request.param)
    file_contents = filename.read_text(encoding="utf-8")
    contents = json.loads(file_contents)
    executable = [
        "### Prevent plots from rendering for profiling. ###\n",
        "from unittest.mock import patch, MagicMock\n",
        "from pathlib import Path\n",
        "patch_show = patch('matplotlib.pyplot.show', new=MagicMock())\n",
        "patch_show.start()\n",
        "patch_save = patch('matplotlib.pyplot.savefig', new=MagicMock())\n",
        "patch_save.start()\n",
        "patch_cwd = patch('pathlib.Path.cwd', new=MagicMock(return_value=Path(__file__).parent))\n",
        "patch_cwd.start()\n",
    ]
    for cell in contents["cells"]:
        if cell["cell_type"] == "code":
            code = [line for line in cell["source"] if line[0] not in {"%", "!"}]
            if len(code) > 0 and '#include "rebound.h"' in code[0]:
                continue
            executable.extend(code + ["\n"])
    executable.extend(
        [
            "patch_cwd.stop()\n",
            "patch_show.stop()\n",
            "patch_save.stop()\n",
        ]
    )
    shutil.copytree(filename.parent, tmp_path, dirs_exist_ok=True)
    output_file = tmp_path / filename.with_suffix(".py").name
    output_file.write_text("".join(executable), encoding="utf-8")
    return output_file


@pytest.mark.slow
@pytest.mark.parametrize(
    "convert_ipynb_to_py", EXAMPLES, ids=[f.name for f in EXAMPLES], indirect=True
)
def test_notebooks(convert_ipynb_to_py):
    """Convert a notebook to a script and test it."""
    output = subprocess.run(
        [str(sys.executable), str(convert_ipynb_to_py)],
        shell=False,
        capture_output=True,
        check=False,
        text=True,
    )
    assert output.returncode == 0, print("\n", output.stdout, output.stderr)
