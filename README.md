# Clash Royale Descriptive Analytics (Nov 2022 – Mar 2023)

An end-to-end PySpark pipeline over five seasons of high-trophy Clash Royale battles (149 daily files, 193.9 M raw rows). It produces descriptive metrics for cards, decks and deck families.

## What to submit: `submission/`

Everything except the raw data is in this folder:

| Path | What it is |
|---|---|
| `submission/clash_royale_descriptive_analysis.ipynb` | **The technical report**: an executed notebook (abstract, methodology, results, sensitivity analysis, conclusion, references) |
| `submission/outputs/du_sch_raw_input.png` | `du -sch` capture of the 38 GB raw input |
| `submission/outputs/fig1–fig7*.png` | The seven report figures at 300 DPI |
| `submission/outputs/*.parquet` | Result tables: trophy thresholds, card metrics, deck leaderboard, family summary, cross-season stability, trajectories, ingestion audits, filtering funnel, sensitivity results |
| `submission/ref/cards.json` | Card id → name / elixir / rarity lookup used by the notebook |

## Development files (not for submission)

| Path | What it is |
|---|---|
| `notebook_src.py` | Source of the notebook in percent format; this is the file to edit |
| `build_notebook.py` | Converts `notebook_src.py` into `submission/clash_royale_descriptive_analysis.ipynb` |

## Not included

These folders are rebuilt by running the pipeline and are too large for GitHub:

- `submission/outputs/staged_battles.parquet` (about 9 GB): the 191 M cleaned battles
- `submission/outputs/_interim/` (about 8 GB): intermediate tables

## Reproduce

Requires PySpark 4.x, Java 17 and the raw data at `/mnt/data/public/clash-royale-games`.

```bash
python build_notebook.py
cd submission
jupyter nbconvert --to notebook --execute --inplace clash_royale_descriptive_analysis.ipynb
```

| Environment variable | Effect |
|---|---|
| `CR_SMOKE=1` | Quick test run on a few files |
| `CR_REBUILD=1` | Forces a full rebuild, about 40 min on 4 CPUs / 16 GiB |
