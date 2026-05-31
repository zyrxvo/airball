"""Shared utilities for the AIRBALL example notebooks.

This module is imported by the notebooks in docs/examples/ and is intentionally
excluded from the built documentation (see the 'exclude' plugin in mkdocs.yml).

Important: notebooks must be run with their working directory set to
docs/examples/ so that Path.cwd()-relative image paths resolve correctly both
during local execution and MkDocs builds.  Do not import this module from
outside the docs/examples/ directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Markdown


def save_and_display_figure(filepath: Path, alt_text: str, width: int = 600) -> Markdown:
    """Save the current matplotlib figure to filepath and return it as a markdown image.

    Uses plt.gcf() to access the current figure, so call this while the figure
    is still active (before any subsequent plt.subplots() or plt.close() call).
    """
    plt.savefig(
        filepath,
        dpi=192,
        facecolor=plt.gcf().get_facecolor(),
        edgecolor="none",
        pad_inches=0.25,
        bbox_inches="tight",
    )
    plt.close()
    img = str(filepath.relative_to(Path.cwd()))
    return Markdown(f"""<img src="{img}" alt="{alt_text}" width="{width}">""")
