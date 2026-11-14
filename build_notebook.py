"""Convert the percent-format source (notebook_src.py) into outputs/clash_royale_descriptive_analysis.ipynb.

Cells are delimited by lines starting with '# %%'. '# %% [markdown]' cells have their
leading '# ' comment prefix stripped.
"""
import re
import sys
from pathlib import Path

import nbformat

src = Path(sys.argv[1] if len(sys.argv) > 1 else "notebook_src.py").read_text()
out = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/clash_royale_descriptive_analysis.ipynb")

cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    body = body.strip("\n")
    if "[markdown]" in header:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    elif body.strip():
        cells.append(nbformat.v4.new_code_cell(body))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
