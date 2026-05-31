import re
from pathlib import Path


def on_config(config, **kwargs):
    """Keep the docs homepage in sync with README.md.

    README.md is the canonical home page for both GitHub and the docs site.
    This hook copies it to docs/index.md before MkDocs collects files.
    """
    repo_root = Path(config["docs_dir"]).parent
    readme = repo_root / "README.md"
    index = Path(config["docs_dir"]) / "index.md"
    if readme.exists():
        index.write_text(readme.read_text(encoding="utf-8"), encoding="utf-8")
    return config


def on_page_content(html, page, config, **kwargs):
    """Adjust relative image src paths in notebook-generated pages.

    When use_directory_urls=true (the default), mkdocs-jupyter serves
    docs/examples/foo.ipynb at examples/foo/index.html — one virtual directory
    level deeper than the source file's location. A bare relative path like
    'images/foo.png' therefore resolves to examples/foo/images/foo.png on the
    built site instead of the correct examples/images/foo.png.

    This hook fixes that by prepending '../' to bare relative <img src> values
    on notebook pages. Two scoping decisions keep it safe:

    1. It only runs when use_directory_urls=true (the MkDocs setting that
       causes the depth shift). If that setting is ever disabled, the hook
       becomes a no-op automatically.
    2. It only rewrites paths on pages sourced from .ipynb files; plain
       Markdown pages are left untouched.

    The regex is idempotent: paths already starting with '../' are excluded
    by the negative lookahead, so running the hook twice on the same content
    is safe.
    """
    if not config.get("use_directory_urls", True):
        return html

    if not page.file.src_path.endswith(".ipynb"):
        return html

    # Prepend '../' to img src values that are bare relative paths,
    # i.e. not already starting with http(s):, data:, /, or ../
    # The negative lookahead for '../' is the idempotency guard — if a
    # notebook author has already corrected a path manually, it won't be
    # double-prefixed.
    html = re.sub(
        r'(<img\b[^>]*\bsrc=")(?!https?:|data:|/|\.\./)([^"]*")',
        r"\1../\2",
        html,
    )
    return html
