import re


def on_page_content(html, page, **kwargs):
    """Adjust relative image src paths in notebook-generated pages.

    mkdocs-jupyter serves docs/examples/foo.ipynb at examples/foo/ (one level
    deeper due to use_directory_urls=true). A bare relative path like
    'images/foo.png' works correctly in local Jupyter (where cwd is the
    notebook's parent directory) but resolves one level too deep on the site.
    This hook prepends '../' to any such paths so they resolve correctly for
    both contexts.
    """
    if not page.file.src_path.endswith(".ipynb"):
        return html

    # Prepend '../' to img src values that are bare relative paths,
    # i.e. not already starting with http(s):, data:, /, or ../
    html = re.sub(
        r'(<img\b[^>]*\bsrc=")(?!https?:|data:|/|\.\./)([^"]*")',
        r"\1../\2",
        html,
    )
    return html
