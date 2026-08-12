"""Convert a Jupyter notebook to a runnable Python script."""

import json
import sys
from pathlib import Path


def convert(notebook_path: Path, output_path: Path) -> None:
    contents = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_dir = notebook_path.parent.resolve()

    executable = [
        "### Prevent plots from rendering. ###\n",
        "from unittest.mock import patch, MagicMock\n",
        "from pathlib import Path\n",
        "patch_show = patch('matplotlib.pyplot.show', new=MagicMock())\n",
        "patch_show.start()\n",
        "patch_save = patch('matplotlib.pyplot.savefig', new=MagicMock())\n",
        "patch_save.start()\n",
        f"patch_cwd = patch('pathlib.Path.cwd', new=MagicMock(return_value=Path(r'{notebook_dir}')))\n",
        "patch_cwd.start()\n",
    ]
    for cell in contents["cells"]:
        if cell["cell_type"] == "code":
            code = [line for line in cell["source"] if line and line[0] not in {"%", "!"}]
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
    output_path.write_text("".join(executable), encoding="utf-8")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
