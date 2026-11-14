# Clash Royale Descriptive Analytics (Nov 2022 – Mar 2023)

An end-to-end PySpark pipeline over five seasons of high-trophy Clash Royale battles (149 daily files, 193.9 M raw rows). It produces descriptive metrics for cards, decks and deck families.

## Contents

| Path | What it is |
|---|---|
| `outputs/clash_royale_descriptive_analysis.ipynb` | Executed notebook: the main deliverable, including the written interpretation |
| `outputs/fig*.png` | Five figures at 300 DPI |
| `outputs/*.parquet` | Result tables (trophy thresholds, card metrics, deck leaderboard, family summary, cross-season stability, trajectories, ingestion audits) |
| `notebook_src.py` | Source of the notebook (percent format), the single source of truth |
| `build_notebook.py` | Converts `notebook_src.py` into the `.ipynb` |
| `ref/cards.json` | Card id → name / elixir / rarity lookup |

## Not included

These folders are rebuilt by running the pipeline and are too large for GitHub:

- `outputs/staged_battles.parquet` (about 9 GB): the 191 M cleaned battles
- `outputs/_interim/` (about 8 GB): intermediate tables

## Reproduce

Requires PySpark 4.x, Java 17 and the raw data at `/mnt/data/public/clash-royale-games`.

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace outputs/clash_royale_descriptive_analysis.ipynb
```

| Environment variable | Effect |
|---|---|
| `CR_SMOKE=1` | Quick test run on a few files |
| `CR_REBUILD=1` | Forces a full rebuild, about 40 min on 4 CPUs / 16 GiB |
