# %% [markdown]
# # Card and Deck Performance in High-Trophy Clash Royale Battles Across Seasons
#
# **A descriptive big-data analysis with Apache Spark** of five consecutive full seasons
# (8 Nov 2022 – 3 Apr 2023) of the Clash Royale battle logs on Jojie
# (`/mnt/data/public/clash-royale-games`, read-only).
#
# **Research question.** Do the cards and decks used most often in high-trophy play (highest pick rate)
# also win broadly (high win rate across many players)? Or do one-trick specialists,
# short-lived meta anomalies, or small samples skew the leaderboard?
#
# The analysis is strictly **descriptive**. Win rates describe outcomes observed in this sample.
# They are not causal estimates of card strength or player skill.
#
# | Section | Content | Gate |
# |---|---|---|
# | 1 | Environment, Spark session, constants | – |
# | 2 | Ingestion, schema enforcement, deduplication → `staged_battles.parquet` | A |
# | 3 | Mode filtering & per-season trophy thresholds τ_s | B |
# | 4 | Deck canonicalisation & deck families | C |
# | 5 | Card / deck / family metrics; breadth vs. specialisation | D |
# | 6 | Cross-season stability & balance-patch trajectories | E |
# | 7 | Figures 1–5 | – |
# | 8 | Deliverable audit | F |
# | 9 | Findings & interpretation | – |

# %% [markdown]
# ## 1. Environment & configuration
#
# **Compute profile.** Resources are read from the container's cgroup, not from `nproc`/`free`.
# On Jojie those report the host (32 cores, ~220 GB), but this pod is limited to **4 CPUs and 16 GiB**.
# The spec's `local[*]` with a 128 GB driver would run 32 tasks on 4 cores inside a heap larger than
# the container, and the kernel would OOM-kill it. Spark therefore runs `local[<cpu quota>]` with a heap
# of about 50% of the memory limit. Nothing large is cached in memory, and every heavy stage is
# materialised as Parquet. Each stage writes a `_SUCCESS` marker and is skipped on re-run, so an
# interrupted run resumes where it stopped (`CR_REBUILD=1` forces full recomputation). Intermediate
# tables live in `outputs/_interim/` on persistent storage. Only shuffle spill uses node-local `/tmp`.

# %%
import json, math, os, shutil, time, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.text
from scipy import stats

from pyspark.sql import SparkSession, functions as F, types as T, Window

T0 = time.time()
PROJECT_DIR = Path.cwd() if Path.cwd().name != "outputs" else Path.cwd().parent
SMOKE = os.environ.get("CR_SMOKE", "0") == "1"      # smoke test: a few files per season, separate output dir
OUT = PROJECT_DIR / ("outputs_smoke" if SMOKE else "outputs")
OUT.mkdir(exist_ok=True)
SHUFFLE_DIR = Path(os.environ.get("CR_SHUFFLE_DIR", "/tmp/spark-cr/local"))   # node-local disk: shuffle spill only
SHUFFLE_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = OUT / "_interim"                                           # persistent intermediate parquet (survives restarts)
SCRATCH.mkdir(exist_ok=True)
# CR_REBUILD=1 forces every stage to recompute. Default: resume, i.e. skip any stage whose output has a _SUCCESS marker.
REBUILD = os.environ.get("CR_REBUILD", "0") == "1"

def done(path):
    return (Path(path) / "_SUCCESS").exists() and not REBUILD

def cgroup_limits():
    """Container CPU quota and memory limit (cgroup v2). Host-level nproc/free overstate what this pod may use."""
    cpus, mem = os.cpu_count(), None
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            cpus = max(1, int(int(q) / int(per)))
    except Exception:
        pass
    try:
        m = open("/sys/fs/cgroup/memory.max").read().strip()
        mem = None if m == "max" else int(m)
    except Exception:
        pass
    return cpus, mem
CPUS, MEM_LIMIT = cgroup_limits()
# JVM heap ~50% of the container limit: leaves room for JVM off-heap, the Python kernel and pandas frames.
DRIVER_GB = int(os.environ.get("CR_DRIVER_GB", max(4, int((MEM_LIMIT or 64 * 2**30) * 0.5 / 2**30))))
print(f"container: {CPUS} CPUs, memory limit {MEM_LIMIT / 2**30 if MEM_LIMIT else float('nan'):.0f} GiB -> driver heap {DRIVER_GB}g")

RAW_ROOT = "/mnt/data/public/clash-royale-games"
SEASONS = {  # season_id: (folder, first battle date, last battle date) - dates inclusive, UTC
    "S01": ("20221108-20221205", "2022-11-08", "2022-12-05"),
    "S02": ("20221206-20230102", "2022-12-06", "2023-01-02"),
    "S03": ("20230103-20230206", "2023-01-03", "2023-02-05"),
    "S04": ("20230206-20230306", "2023-02-06", "2023-03-05"),
    "S05": ("20230306-20230403", "2023-03-06", "2023-04-03"),
}
SEASON_LABEL = {"S01": "Nov 2022", "S02": "Dec 2022", "S03": "Jan 2023", "S04": "Feb 2023", "S05": "Mar 2023"}

MODE_NAMES = {72000006: "Ladder", 72000044: "Ladder_GoldRush", 72000066: "Showdown_Ladder",
              72000323: "Ranked1v1 (PoL)", 72000327: "Ranked1v1_GoldRush",
              72000291: "Challenge_AllCards_EventDeck", 72000007: "Friendly", 72000009: "Tournament"}
COLLECTION_MODES = [72000006, 72000044, 72000066, 72000323, 72000327]   # deckSelection == 'Collection'
LADDER_MODES = [72000006, 72000044]                                      # used to profile tau_s

TAU_QUANTILE = 0.75        # tau_s = P75 of the season's ladder starting-trophy distribution
EB_MIN_N = 100             # decks/cards with >= this many non-mirror matches are used to fit the Beta prior
EB_FALLBACK_M = 200        # pseudo-observation strength if the moment fit is degenerate
MIN_PLAYER_GAMES = 5       # player-weighted WR uses players with >= 5 non-mirror games on the deck
LEADERBOARD_MIN_OBS = 50   # decks kept in deck_leaderboard.parquet (per scope)
CLASS_MIN_MATCHES = 200    # decks eligible for specialist / staple classification
CLASS_MIN_PLAYERS = 10     #   ... and with >= 10 qualifying players
SPEC_GAP = 0.03            # specialist gap threshold (+3.0 pp)
FAMILY_SEED_MIN_OBS = 20   # decks with >= 20 pooled observations take part in greedy family-head formation
FOCUS_CARDS = {26000087: "Phoenix", 26000077: "Monk", 28000002: "Rage"}

# Palette (validated reference instance - first three categorical slots are all-pairs safe)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
RAMP = {"250": "#86b6ef", "400": "#3987e5", "600": "#184f95"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "savefig.dpi": 300, "savefig.bbox": "tight",
})

spark = (SparkSession.builder.appName("clash-royale-descriptive")
         .master(f"local[{CPUS}]")
         .config("spark.driver.memory", f"{DRIVER_GB}g")
         .config("spark.driver.maxResultSize", f"{max(1, DRIVER_GB // 4)}g")
         .config("spark.sql.shuffle.partitions", 200)
         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
         .config("spark.sql.files.maxPartitionBytes", str(64 * 2**20))
         .config("spark.memory.fraction", "0.6")
         .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
         .config("spark.sql.autoBroadcastJoinThreshold", str(32 * 2**20))
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.local.dir", str(SHUFFLE_DIR))
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("Spark", spark.version, "| default parallelism", spark.sparkContext.defaultParallelism,
      "| output dir", OUT, "| rebuild", REBUILD)

def to_pd(sdf):
    return sdf.toPandas()

def declutter(ax, texts, step=2.0, iters=150, max_shift=28):
    """Nudge annotation offsets vertically until their rendered boxes no longer overlap."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for _ in range(iters):
        # text box only - an Annotation's own extent includes its leader line, which grows as it moves
        bbs = [matplotlib.text.Text.get_window_extent(t, r).expanded(1.04, 1.12) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if bbs[i].overlaps(bbs[j]):
                    lo, hi = (i, j) if bbs[i].y0 <= bbs[j].y0 else (j, i)
                    for k, d in ((lo, -step), (hi, step)):
                        x, y = texts[k].xyann
                        texts[k].xyann = (x, float(np.clip(y + d, -max_shift, max_shift)))
                    moved = True
        if not moved:
            break
        fig.canvas.draw()

def save_parquet_pd(pdf, name):
    path = OUT / name
    pdf.to_parquet(path, index=False)
    return path

# %% [markdown]
# ### Card lookup
# Card IDs are official Supercell IDs. Names, elixir cost, type and rarity come from the public
# RoyaleAPI `cards.json` (cached under `ref/` for reproducibility). The table is broadcast for joins.

# %%
CARDS_JSON = PROJECT_DIR / "ref" / "cards.json"
if not CARDS_JSON.exists():
    import urllib.request
    CARDS_JSON.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve("https://royaleapi.github.io/cr-api-data/json/cards.json", CARDS_JSON)
cards_pd = pd.DataFrame(json.load(open(CARDS_JSON)))[["id", "name", "elixir", "type", "rarity"]]
cards_pd = cards_pd.rename(columns={"id": "card_id", "name": "card_name"})
CARD_NAME = dict(zip(cards_pd.card_id, cards_pd.card_name))
cards_sdf = F.broadcast(spark.createDataFrame(cards_pd))
print(len(cards_pd), "cards in lookup")

# %% [markdown]
# ## 2. Ingestion, schema enforcement & deduplication (Gate A)
#
# **Directory layout:** `<root>/<season>/<season>/<YYYYMMDD>.csv`. The CSVs are headerless with 24 columns.
# Only the five in-scope inner season folders are read.
#
# **Observed data issue: overlapping boundary files.** `20230206.csv` exists in both the S03 and S04
# folders, and `20230306.csv` in both S04 and S05. The two copies are near-identical. All 149 files
# are ingested. Each battle gets its `season_id` from its **battle date**, not its folder. Global
# deduplication then collapses the duplicated boundary days. A season's boundary day belongs to the
# season that starts on it. Seasons reset on Mondays, so day granularity is used.
#
# **Deduplication** is a single shuffle:
# 1. Canonicalise orientation so that `p1_tag < p2_tag` (swap all player fields if not). A swapped
#    perspective copy (P2 vs P1 with mirrored crowns) then becomes identical to its partner.
# 2. Group by the canonical key `(battle_timestamp, game_mode_id, min(tag), max(tag))`. Keep one
#    deterministic record per group (`min` over the record struct).
#    `battle_id = xxhash64(timestamp, mode, min_tag, max_tag)` is stored as a compact identifier. Dedup
#    itself uses the exact key, so hash collisions cannot merge battles.
# 3. Each group records how many raw rows, distinct raw lines and distinct orientations it had.
#    The audit separates *exact duplicate lines* from *perspective duplicates*.
#
# **Validity filter (applied before dedup):** all 24 fields must parse, crowns must be in 0–3, and
# each deck must hold 8 distinct cards.

# %%
p_fields = ([("battle_timestamp_raw", T.StringType()), ("game_mode_id", T.IntegerType()),
             ("p1_tag", T.StringType()), ("p1_trophies", T.IntegerType()), ("p1_crowns", T.IntegerType())]
            + [(f"p1_c{i}", T.IntegerType()) for i in range(1, 9)]
            + [("p2_tag", T.StringType()), ("p2_trophies", T.IntegerType()), ("p2_crowns", T.IntegerType())]
            + [(f"p2_c{i}", T.IntegerType()) for i in range(1, 9)])
RAW_SCHEMA = T.StructType([T.StructField(n, t, True) for n, t in p_fields])
assert len(RAW_SCHEMA) == 24

season_paths = {sid: f"{RAW_ROOT}/{folder}/{folder}/*.csv" for sid, (folder, _, _) in SEASONS.items()}
files_on_disk = {sid: sorted(glob.glob(p)) for sid, p in season_paths.items()}
n_files = sum(len(v) for v in files_on_disk.values())
print({sid: len(v) for sid, v in files_on_disk.items()}, "total", n_files)
assert n_files == 149, n_files
N_EXPECTED_FILES = 149
read_paths = list(season_paths.values())
if SMOKE:   # one mid-season file per season + both copies of the duplicated boundary day 20230206
    read_paths = sorted({v[len(v) // 2] for v in files_on_disk.values()}
                        | {f for v in files_on_disk.values() for f in v if f.endswith("20230206.csv")})
    N_EXPECTED_FILES = len(read_paths)
    print("SMOKE TEST on", N_EXPECTED_FILES, "files")

raw = (spark.read.schema(RAW_SCHEMA).option("header", "false").option("mode", "PERMISSIVE")
       .csv(read_paths)
       .withColumn("source_file", F.input_file_name())
       .withColumn("source_folder", F.regexp_extract("source_file", r"clash-royale-games/([0-9-]+)/", 1)))

p1_cards = F.array(*[F.col(f"p1_c{i}") for i in range(1, 9)])
p2_cards = F.array(*[F.col(f"p2_c{i}") for i in range(1, 9)])
raw = (raw
       .withColumn("battle_timestamp", F.to_timestamp("battle_timestamp_raw", "yyyyMMdd'T'HHmmss.SSS'Z'"))
       .withColumn("p1_cards", p1_cards).withColumn("p2_cards", p2_cards)
       .withColumn("battle_date", F.to_date("battle_timestamp")))

season_expr = F.lit(None).cast("string")
for sid, (_, d0, d1) in SEASONS.items():
    season_expr = F.when(F.col("battle_date").between(F.lit(d0).cast("date"), F.lit(d1).cast("date")),
                         F.lit(sid)).otherwise(season_expr)
raw = raw.withColumn("season_id", season_expr)

all_fields_present = F.lit(True)
for n, _ in p_fields:
    all_fields_present = all_fields_present & F.col(n).isNotNull()
is_valid = (all_fields_present & F.col("battle_timestamp").isNotNull()
            & F.col("p1_crowns").between(0, 3) & F.col("p2_crowns").between(0, 3)
            & (F.size(F.array_distinct("p1_cards")) == 8) & (F.size(F.array_distinct("p2_cards")) == 8)
            & (F.col("p1_tag") != F.col("p2_tag")))
raw = raw.withColumn("is_valid", is_valid)

# %%
# Stage 2a: map-only pass, CSV -> typed parquet partitioned by battle-date season (no shuffle, bounded memory).
RAW_PQ = SCRATCH / "raw_typed.parquet"
if not done(RAW_PQ):
    t = time.time()
    (raw.withColumn("source_file", F.regexp_extract("source_file", r"([0-9]{8}\.csv)$", 1))
        .withColumn("season_id", F.coalesce("season_id", F.lit("OUT_OF_WINDOW")))
        .drop("battle_timestamp_raw", "p1_cards", "p2_cards", "battle_date")
        .write.mode("overwrite").partitionBy("season_id").parquet(str(RAW_PQ)))
    print(f"CSV -> parquet {time.time()-t:.0f}s")
raw_pq = spark.read.parquet(str(RAW_PQ))

FILE_AUDIT = OUT / "ingestion_file_audit.parquet"
file_audit = to_pd(raw_pq.groupBy("source_folder", "source_file")
                   .agg(F.count("*").alias("raw_rows"),
                        F.sum((~F.col("is_valid")).cast("long")).alias("invalid_rows"),
                        F.sum((F.col("season_id") == "OUT_OF_WINDOW").cast("long")).alias("out_of_window_rows"),
                        F.min("battle_timestamp").alias("min_ts"), F.max("battle_timestamp").alias("max_ts")))
file_audit = file_audit.rename(columns={"source_file": "file"}).sort_values(["source_folder", "file"])
file_audit.to_parquet(FILE_AUDIT, index=False)
assert len(file_audit) == N_EXPECTED_FILES, "every CSV file must be registered by the reader"
folder_audit = (file_audit.groupby("source_folder")
                .agg(files=("file", "count"), raw_rows=("raw_rows", "sum"), invalid_rows=("invalid_rows", "sum"),
                     out_of_window_rows=("out_of_window_rows", "sum")).reset_index())
folder_audit

# %%
# Stage 2b: deduplication, one season at a time (each shuffle ~1/5 of the data).
# A battle's season is fixed by its date, so duplicate copies (boundary-day files in two folders,
# swapped perspectives) always land in the same season partition.
STAGED = OUT / "staged_battles.parquet"
p1_cards = F.array(*[F.col(f"p1_c{i}") for i in range(1, 9)])
p2_cards = F.array(*[F.col(f"p2_c{i}") for i in range(1, 9)])
swap = F.col("p1_tag") > F.col("p2_tag")
def pick(a, b):
    return F.when(swap, b).otherwise(a)
KEY = ["battle_timestamp", "game_mode_id", "p1_tag", "p2_tag"]
content = F.struct("p1_trophies", "p1_crowns", "p1_cards", "p2_trophies", "p2_crowns", "p2_cards")

def stage_season(sid):
    src = raw_pq.filter((F.col("season_id") == sid) & F.col("is_valid"))
    raw_cols = ["battle_timestamp", "game_mode_id", "p1_tag", "p1_trophies", "p1_crowns"] + [f"p1_c{i}" for i in range(1, 9)] \
               + ["p2_tag", "p2_trophies", "p2_crowns"] + [f"p2_c{i}" for i in range(1, 9)]
    canon = (src.withColumn("raw_line_hash", F.xxhash64(*raw_cols))
             .withColumn("p1_cards", p1_cards).withColumn("p2_cards", p2_cards)
             .select("battle_timestamp", "game_mode_id", "raw_line_hash", swap.cast("int").alias("was_swapped"),
                     pick(F.col("p1_tag"), F.col("p2_tag")).alias("p1_tag"), pick(F.col("p1_trophies"), F.col("p2_trophies")).alias("p1_trophies"),
                     pick(F.col("p1_crowns"), F.col("p2_crowns")).alias("p1_crowns"), pick(F.col("p1_cards"), F.col("p2_cards")).alias("p1_cards"),
                     pick(F.col("p2_tag"), F.col("p1_tag")).alias("p2_tag"), pick(F.col("p2_trophies"), F.col("p1_trophies")).alias("p2_trophies"),
                     pick(F.col("p2_crowns"), F.col("p1_crowns")).alias("p2_crowns"), pick(F.col("p2_cards"), F.col("p1_cards")).alias("p2_cards")))
    # step 1: collapse identical raw lines (plain count/first aggregates - no collect_set buffers)
    lines = canon.groupBy(*KEY, "raw_line_hash").agg(
        F.count("*").alias("n"), F.first("was_swapped").alias("was_swapped"), F.first(content).alias("r"))
    # step 2: collapse remaining copies of the same canonical battle (swapped perspectives / conflicting copies)
    return (lines.groupBy(*KEY).agg(
                F.sum("n").alias("dup_raw_rows"), F.count("*").alias("dup_distinct_lines"),
                (F.max("was_swapped") - F.min("was_swapped") + 1).alias("dup_orientations"),
                F.when(F.min(F.xxhash64("r")) != F.max(F.xxhash64("r")), 2).otherwise(1).alias("dup_contents"),
                F.min("r").alias("r"))
            .select(F.xxhash64(*KEY).alias("battle_id"), "battle_timestamp", "game_mode_id",
                    "p1_tag", "r.p1_trophies", "r.p1_crowns", "r.p1_cards",
                    "p2_tag", "r.p2_trophies", "r.p2_crowns", "r.p2_cards",
                    "dup_raw_rows", "dup_distinct_lines", "dup_orientations", "dup_contents"))

for sid in SEASONS:
    part = STAGED / f"season_id={sid}"
    if done(part):
        print(sid, "staged (cached)"); continue
    t = time.time()
    stage_season(sid).write.mode("overwrite").parquet(str(part))
    print(f"{sid} dedup + staging {time.time()-t:.0f}s")
staged = spark.read.parquet(str(STAGED))
staged.printSchema()

# %%
dedup_audit = to_pd(staged.groupBy("season_id").agg(
    F.sum("dup_raw_rows").alias("valid_rows_in"),
    F.sum(F.col("dup_raw_rows") - F.col("dup_distinct_lines")).alias("exact_duplicates_removed"),
    F.sum(F.col("dup_distinct_lines") - 1).alias("perspective_or_conflict_duplicates_removed"),
    F.sum((F.col("dup_orientations") > 1).cast("long")).alias("battles_seen_in_both_orientations"),
    F.sum((F.col("dup_contents") > 1).cast("long")).alias("battles_with_conflicting_copies"),  # copies differing in content
    F.count("*").alias("clean_battles"))).sort_values("season_id").reset_index(drop=True)
assert (dedup_audit.valid_rows_in - dedup_audit.exact_duplicates_removed
        - dedup_audit.perspective_or_conflict_duplicates_removed == dedup_audit.clean_battles).all()
total_raw = int(file_audit.raw_rows.sum())
print(f"raw rows ingested: {total_raw:,} | invalid: {int(file_audit.invalid_rows.sum()):,} | "
      f"out-of-window: {int(file_audit.out_of_window_rows.sum()):,} | clean battles: {int(dedup_audit.clean_battles.sum()):,}")
dedup_audit.to_parquet(OUT / "ingestion_dedup_audit.parquet", index=False)
dedup_audit

# %% [markdown]
# ## 3. Mode filtering & empirical seasonal trophy profiling (Gate B)
#
# First, the mode mix of the clean data per season:

# %%
mode_mix = to_pd(staged.groupBy("season_id", "game_mode_id").count())
mode_mix["mode"] = mode_mix.game_mode_id.map(MODE_NAMES).fillna("other")
mode_mix["collection"] = mode_mix.game_mode_id.isin(COLLECTION_MODES)
mode_tab = mode_mix.pivot_table(index=["game_mode_id", "mode", "collection"], columns="season_id",
                                values="count", fill_value=0).astype(int)
mode_tab

# %% [markdown]
# **Trophy field semantics differ by mode.** The ladder modes record ladder trophies (capped at 7,500).
# In **Path of Legends (72000323)** the same field holds Path-of-Legends league points: most values
# are 0–30, and only top-league players reach ~1,000+. These are not on the ladder scale. Requiring
# both players to be ≥ τ_s (a ladder-trophy threshold) therefore removes PoL battles by construction.
# This is not an extra exclusion. The high-trophy sample is in practice **high-trophy Trophy Road**
# (Ladder, Ladder Gold Rush, Showdown Ladder). Summary statistics below confirm this.

# %%
pol_trophies = to_pd(staged.filter("game_mode_id = 72000323")
                     .select(F.explode(F.array("p1_trophies", "p2_trophies")).alias("t"))
                     .agg(F.expr("percentile_approx(t, array(0.5, 0.75, 0.9, 0.99, 0.999))").alias("q"),
                          F.max("t").alias("max")))
print("PoL 'trophies' field quantiles P50/P75/P90/P99/P99.9:", pol_trophies.q[0], "max", pol_trophies["max"][0])

ladder_obs = (staged.filter(F.col("game_mode_id").isin(LADDER_MODES))
              .select("season_id", F.explode(F.array("p1_trophies", "p2_trophies")).alias("trophies")))
qs = [0.25, 0.5, 0.75, 0.8, 0.9]
thr = to_pd(ladder_obs.groupBy("season_id").agg(
    F.count("*").alias("ladder_player_obs"),
    F.expr(f"percentile_approx(trophies, array({','.join(map(str, qs))}), 100000)").alias("q"),
    F.mean("trophies").alias("mean"), F.max("trophies").alias("max"),
    F.mean((F.col("trophies") >= 7500).cast("double")).alias("share_at_cap"))).sort_values("season_id")
for q, i in zip(["p25", "p50", "p75", "p80", "p90"], range(len(qs))):
    thr[q] = thr.q.map(lambda a: int(a[i]))
thr = thr.drop(columns="q")
thr["tau"] = thr["p75"]
TAU = dict(zip(thr.season_id, thr.tau))
print("tau_s:", TAU)

tau_sdf = F.broadcast(spark.createDataFrame(thr[["season_id", "tau"]]))
coll = staged.filter(F.col("game_mode_id").isin(COLLECTION_MODES)).join(tau_sdf, "season_id")
hi = coll.filter((F.col("p1_trophies") >= F.col("tau")) & (F.col("p2_trophies") >= F.col("tau")))

# How many collection battles qualify, and how a naive static threshold (>= 7,000) would behave instead
qual = to_pd(coll.groupBy("season_id").agg(
    F.count("*").alias("collection_battles"),
    F.sum(F.col("game_mode_id").isin(LADDER_MODES).cast("long")).alias("ladder_battles"),
    F.sum(((F.col("p1_trophies") >= F.col("tau")) & (F.col("p2_trophies") >= F.col("tau"))).cast("long")).alias("high_trophy_battles"),
    F.sum((F.col("game_mode_id").isin(LADDER_MODES) & (F.col("p1_trophies") >= F.col("tau"))
           & (F.col("p2_trophies") >= F.col("tau"))).cast("long")).alias("high_trophy_ladder_battles"),
    F.sum((F.col("game_mode_id").isin(LADDER_MODES) & (F.col("p1_trophies") >= 7000)
           & (F.col("p2_trophies") >= 7000)).cast("long")).alias("static7000_ladder_battles")))
thr = thr.merge(qual, on="season_id")
thr["share_ladder_battles_qualifying_tau"] = thr.high_trophy_ladder_battles / thr.ladder_battles
thr["share_ladder_battles_qualifying_static7000"] = thr.static7000_ladder_battles / thr.ladder_battles
thr["season_label"] = thr.season_id.map(SEASON_LABEL)
thr["tau_quantile"] = TAU_QUANTILE
save_parquet_pd(thr, "seasonal_trophy_thresholds.parquet")
thr

# %%
# Gate B assertions
hi_modes = to_pd(hi.groupBy("game_mode_id").count())
hi_modes["mode"] = hi_modes.game_mode_id.map(MODE_NAMES)
print(hi_modes.to_string(index=False))
assert set(hi_modes.game_mode_id) <= set(COLLECTION_MODES), "only player-constructed modes may remain"
viol = hi.filter((F.col("p1_trophies") < F.col("tau")) | (F.col("p2_trophies") < F.col("tau"))).count()
assert viol == 0
print("Gate B passed: collection modes only; both players >= tau_s in every retained battle")

# %% [markdown]
# ## 4. Perspective unpivoting, deck canonicalisation & deck families (Gate C)
#
# Each high-trophy battle is unpivoted into two directed observations, one per player.
# `win = crowns_player > crowns_opp` and `margin = |crowns_player − crowns_opp|`.
# **Ties** (equal crowns) are rare (~0.01% of battles). They are **excluded** from every metric, and
# their count is reported. Every pick-rate denominator is `2 × N_battles` over the same non-tie set.
#
# `deck_id = concat_ws('-', sort_array(cards))` makes the deck invariant to slot order.

# %%
OBS = SCRATCH / "observations.parquet"
def side(me, op):
    return F.struct(F.col(f"{me}_tag").alias("player_tag"), F.col(f"{me}_trophies").alias("trophies"),
                    F.col(f"{me}_crowns").alias("crowns"), F.col(f"{op}_crowns").alias("opp_crowns"),
                    F.array_sort(F.col(f"{me}_cards")).alias("cards"), F.array_sort(F.col(f"{op}_cards")).alias("opp_cards"))
DECK_ID = lambda c: F.concat_ws("-", F.transform(F.sort_array(c), lambda x: x.cast("string")))

tie_counts = to_pd(hi.groupBy("season_id").agg(F.count("*").alias("hi_battles_incl_ties"),
                                                F.sum((F.col("p1_crowns") == F.col("p2_crowns")).cast("long")).alias("ties")))
if not done(OBS):
    t = time.time()
    obs_w = (hi.filter(F.col("p1_crowns") != F.col("p2_crowns"))
             .select("season_id", "battle_id", "battle_timestamp", "game_mode_id",
                     F.explode(F.array(side("p1", "p2"), side("p2", "p1"))).alias("o"))
             .select("season_id", "battle_id", "battle_timestamp", "game_mode_id", "o.*")
             .withColumn("week", F.to_date(F.date_trunc("week", "battle_timestamp")))
             .withColumn("win", (F.col("crowns") > F.col("opp_crowns")).cast("int"))
             .withColumn("margin", F.abs(F.col("crowns") - F.col("opp_crowns")))
             .withColumn("deck_id", DECK_ID(F.col("cards")))
             .withColumn("opp_deck_id", DECK_ID(F.col("opp_cards")))
             .withColumn("deck_mirror", (F.col("deck_id") == F.col("opp_deck_id")).cast("int")))
    obs_w.write.mode("overwrite").partitionBy("season_id").parquet(str(OBS))
    print(f"observations written {time.time()-t:.0f}s")
obs = spark.read.parquet(str(OBS))
season_n = to_pd(obs.groupBy("season_id").agg((F.count("*") / 2).cast("long").alias("n_battles"),
                                               F.countDistinct("player_tag").alias("distinct_players")))
season_n = season_n.merge(tie_counts, on="season_id").sort_values("season_id").reset_index(drop=True)
N_BATTLES = dict(zip(season_n.season_id, season_n.n_battles))
N_BATTLES["ALL"] = int(season_n.n_battles.sum())
assert (season_n.hi_battles_incl_ties - season_n.ties == season_n.n_battles).all()
season_n

# %%
# Gate C: permutation invariance on test samples - shuffle slot order and recompute deck_id with the same expression
sample = obs.select("cards", "deck_id").limit(2000).toPandas()
rng = np.random.default_rng(7)
shuffled = spark.createDataFrame(pd.DataFrame({"cards": [list(map(int, rng.permutation(c))) for c in sample.cards],
                                               "expected": sample.deck_id}))
mismatch = shuffled.withColumn("recomputed", DECK_ID(F.col("cards"))).filter("recomputed != expected").count()
assert mismatch == 0
assert obs.filter(F.size(F.split("deck_id", "-")) != 8).count() == 0
print("Gate C: deck_id is permutation invariant on 2,000 shuffled samples; every deck has 8 cards")

# %% [markdown]
# ### Deck families (≥ 7/8 card overlap)
#
# Each deck has eight 7-card *cores* (the deck minus one card). The spec groups decks that share a
# core. Taken transitively (connected components), that rule chains archetypes together:
# A~B share a core, B~C share a different core, and so on. Popular decks that differ in 3–4 cards
# would end up in one giant component (shown below). This pipeline instead uses a
# **non-transitive, volume-ordered greedy assignment**, which keeps the spirit of the ≥ 7/8 rule:
#
# 1. Pool deck volumes across all five seasons and sort decks by volume, descending
#    (decks with ≥ 20 pooled observations).
# 2. A deck that shares a 7-card core with an existing head joins the highest-volume such head.
#    Otherwise it becomes a new **Family Archetype Head** and registers its 8 cores.
#    Only heads register cores, so every member differs from its head by at most one card.
# 3. Low-volume decks (< 20 observations) attach to the highest-volume head they share a core with.
#    If none exists, they form singleton families.
#
# `family_id` is the head's deck_id. By construction it is the most-played variant in its family.

# %%
CORES = lambda c: F.transform(F.sequence(F.lit(0), F.lit(7)),
                              lambda i: F.concat_ws("-", F.transform(F.filter(c, lambda x, j: j != i), lambda x: x.cast("string"))))
DECK_VOL = SCRATCH / "deck_volume.parquet"
if not done(DECK_VOL):
    (obs.groupBy("deck_id").agg(F.count("*").alias("n_obs"), F.first("cards").alias("cards"))
        .withColumn("cores", CORES(F.col("cards"))).write.mode("overwrite").parquet(str(DECK_VOL)))
deck_vol = spark.read.parquet(str(DECK_VOL))
n_decks_total = deck_vol.count()

seed = deck_vol.filter(F.col("n_obs") >= FAMILY_SEED_MIN_OBS).select("deck_id", "n_obs", "cores").toPandas()
seed = seed.sort_values(["n_obs", "deck_id"], ascending=[False, True]).reset_index(drop=True)
core_head, head_of, head_rank = {}, {}, {}
for d, cores in zip(seed.deck_id, seed.cores):
    h = None
    for c in cores:                      # pick the highest-volume head (first registered) among shared cores
        hc = core_head.get(c)
        if hc is not None and (h is None or head_rank[hc] < head_rank[h]):
            h = hc
    if h is None:
        h = d
        head_rank[d] = len(head_rank)
        for c in cores:
            core_head.setdefault(c, d)
    head_of[d] = h
print(f"{n_decks_total:,} distinct decks; {len(seed):,} seed decks -> {len(head_rank):,} family heads")

head_core_sdf = spark.createDataFrame(pd.DataFrame({"core": list(core_head), "head": list(core_head.values()),
                                                    "head_rank": [head_rank[h] for h in core_head.values()]}))
small = deck_vol.filter(F.col("n_obs") < FAMILY_SEED_MIN_OBS).select("deck_id", F.explode("cores").alias("core"))
w = Window.partitionBy("deck_id").orderBy("head_rank")
small_map = (small.join(head_core_sdf, "core").withColumn("rk", F.row_number().over(w))
             .filter("rk = 1").select("deck_id", F.col("head").alias("family_id")))
seed_map = spark.createDataFrame(pd.DataFrame({"deck_id": list(head_of), "family_id": list(head_of.values())}))
family_map = (deck_vol.select("deck_id").join(seed_map.unionByName(small_map), "deck_id", "left")
              .withColumn("family_id", F.coalesce("family_id", "deck_id")))
FAMILY_MAP = SCRATCH / "family_map.parquet"
if not done(FAMILY_MAP):
    family_map.write.mode("overwrite").parquet(str(FAMILY_MAP))
family_map = spark.read.parquet(str(FAMILY_MAP))

# Gate C checks: every deck mapped once; every member within 1 card of its head; head = most played variant
assert family_map.count() == n_decks_total == family_map.select("deck_id").distinct().count()
overlap = (family_map.withColumn("d", F.split("deck_id", "-")).withColumn("h", F.split("family_id", "-"))
           .select(F.size(F.array_intersect("d", "h")).alias("shared")))
assert overlap.filter("shared < 7").count() == 0
fam_vol = family_map.join(deck_vol.select("deck_id", "n_obs"), "deck_id")
top_variant = fam_vol.withColumn("rk", F.row_number().over(Window.partitionBy("family_id").orderBy(F.desc("n_obs"), "deck_id")))
assert top_variant.filter("rk = 1 AND deck_id != family_id").count() == 0
fam_sizes = to_pd(fam_vol.groupBy("family_id").agg(F.count("*").alias("variants"), F.sum("n_obs").alias("n_obs")))
print(f"{len(fam_sizes):,} families; top-100 families cover {fam_sizes.nlargest(100, 'n_obs').n_obs.sum()/fam_sizes.n_obs.sum():.1%} of observations")
print("Gate C: every member shares >= 7 cards with its head; heads are the most-played variant")

# %%
# For contrast: transitive connected components over shared 7-card cores among the seed decks (union-find)
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
first_by_core = {}
for d, cores in zip(seed.deck_id, seed.cores):
    find(d)
    for c in cores:
        o = first_by_core.setdefault(c, d)
        if o != d:
            ra, rb = find(o), find(d)
            if ra != rb: parent[rb] = ra
comp = pd.Series({d: find(d) for d in seed.deck_id})
comp_vol = seed.assign(comp=seed.deck_id.map(comp)).groupby("comp").n_obs.sum().sort_values(ascending=False)
print(f"Transitive components: {comp.nunique():,}; largest component holds "
      f"{comp_vol.iloc[0]/seed.n_obs.sum():.1%} of seed-deck observations "
      f"(greedy head assignment: largest family holds {fam_sizes.n_obs.max()/fam_sizes.n_obs.sum():.1%})")

# %% [markdown]
# ## 5. Performance metrics (Gate D)
#
# * **Pick rate.** Observations containing the card/deck/family, divided by `2 × N_battles` (season, non-tie).
# * **Non-mirror win rate.** For cards, only observations where the opponent does *not* also hold
#   the card. For decks, only those where the opponent's deck is not the identical deck. For families,
#   only those where the opponent's deck is not in the same family.
# * **Empirical-Bayes smoothing.** A Beta(α, β) prior is fitted per season and entity type by
#   method of moments on entities with ≥ 100 non-mirror matches. Binomial sampling variance is
#   subtracted first: `var_true = var(p̂) − mean(p̂(1−p̂)/n)`, `M = μ(1−μ)/var_true − 1`.
#   Then `WR_EB = (wins + M·μ) / (n + M)`. If the fit is degenerate (var_true ≤ 0), μ₀ = 0.5 and M = 200.
# * **Crown margins.** Non-mirror wins split by margin `|crowns − opp_crowns|` ∈ {1, 2, 3}, as the spec defines.
  Margin 1 includes 1–0, 2–1 and 3–2 wins. Margin 3 is only a 3–0 shutout.

# %%
def fit_beta(pdf, n_col="nm_matches", w_col="nm_wins", min_n=EB_MIN_N):
    d = pdf[pdf[n_col] >= min_n]
    if len(d) < 10:
        return 0.5, EB_FALLBACK_M, "fallback (too few entities)"
    p = d[w_col] / d[n_col]
    mu = p.mean()
    var_true = p.var(ddof=1) - (p * (1 - p) / d[n_col]).mean()
    if var_true <= 0:
        return 0.5, EB_FALLBACK_M, "fallback (no excess variance)"
    return float(mu), float(min(max(mu * (1 - mu) / var_true - 1, 1.0), 1e6)), f"moments on {len(d)} entities"

EB_PRIORS = []
def apply_eb(pdf, entity, scope_col="season_id"):
    out = []
    for s, g in pdf.groupby(scope_col):
        mu, M, how = fit_beta(g)
        g = g.copy()
        g["wr_eb"] = (g.nm_wins + M * mu) / (g.nm_matches + M)
        g["eb_prior_mu"], g["eb_prior_M"] = mu, M
        EB_PRIORS.append({"entity": entity, "scope": s, "mu0": mu, "M": M, "alpha": M * mu, "beta": M * (1 - mu), "fit": how})
        out.append(g)
    return pd.concat(out, ignore_index=True)

def add_crowns(pdf):
    for k in (1, 2, 3):
        pdf[f"win_{k}crown_pct"] = np.where(pdf.nm_wins > 0, pdf[f"w{k}"] / pdf.nm_wins.where(pdf.nm_wins > 0), np.nan)
    return pdf

# %% [markdown]
# ### 5a. Card metrics

# %%
card_obs = (obs.select("season_id", "week", "win", "margin", "opp_cards", F.explode("cards").alias("card_id"))
            .withColumn("nm", (~F.array_contains("opp_cards", F.col("card_id"))).cast("int")))
card_week = to_pd(card_obs.groupBy("season_id", "week", "card_id").agg(
    F.count("*").alias("obs"), F.sum("win").alias("wins"),
    F.sum("nm").alias("nm_matches"), F.sum(F.col("nm") * F.col("win")).alias("nm_wins"),
    *[F.sum(F.col("nm") * F.col("win") * (F.col("margin") == k).cast("int")).alias(f"w{k}") for k in (1, 2, 3)]))
week_n = to_pd(obs.groupBy("season_id", "week").agg((F.count("*") / 2).alias("n_battles")))

agg_cols = ["obs", "wins", "nm_matches", "nm_wins", "w1", "w2", "w3"]
card_season = card_week.groupby(["season_id", "card_id"], as_index=False)[agg_cols].sum()
card_all = card_week.groupby("card_id", as_index=False)[agg_cols].sum().assign(season_id="ALL")
cm = pd.concat([card_season, card_all], ignore_index=True)
cm["n_battles"] = cm.season_id.map(N_BATTLES)
cm["pick_rate"] = cm.obs / (2 * cm.n_battles)
cm["wr_all"] = cm.wins / cm.obs
cm["mirror_share"] = 1 - cm.nm_matches / cm.obs
cm["wr_nonmirror"] = cm.nm_wins / cm.nm_matches
cm = add_crowns(apply_eb(cm, "card"))
cm = cm.merge(cards_pd, on="card_id", how="left")
cm["card_name"] = cm.card_name.fillna(cm.card_id.astype(str))
cm["pick_rank"] = cm.groupby("season_id").pick_rate.rank(ascending=False, method="min")
cm["season_label"] = cm.season_id.map(SEASON_LABEL).fillna("All seasons")
# each observation holds exactly 8 cards -> card pick rates sum to 8 per perspective
assert np.allclose(cm.groupby("season_id").pick_rate.sum(), 8.0)
unknown = sorted(set(cm.card_id) - set(cards_pd.card_id))
print("card IDs absent from lookup:", unknown)
save_parquet_pd(cm, "card_seasonal_metrics.parquet")
cm[cm.season_id == "ALL"].sort_values("pick_rate", ascending=False)[
    ["card_name", "pick_rate", "wr_all", "wr_nonmirror", "wr_eb", "mirror_share", "nm_matches"]].head(15)

# %% [markdown]
# ### 5b. Deck metrics, player breadth & specialist gap (Requirement 5)
#
# * `|P_D|`: distinct players who piloted the deck.
# * `WR_match`: non-mirror wins / non-mirror matches.
# * `WR_player`: unweighted mean of per-player non-mirror win rates, over players with ≥ 5 non-mirror
#   games on the deck.
# * `Δ_spec = WR_match − WR_player`. A positive gap means the high-volume pilots win more than the
#   typical pilot.
# * **Classification.** Only decks with ≥ 200 non-mirror matches and ≥ 10 qualifying players, within each scope:
#   * *Specialist-Driven*: Δ_spec > +3.0 pp and |P_D| ≤ the scope median ("moderate player pool").
#   * *Broad Meta Staple*: WR_EB in the top quartile, WR_player ≥ 50%, |P_D| ≥ P75, and |Δ_spec| ≤ 3 pp.
#   * Otherwise *Unclassified*. Decks below the sample floors are *Insufficient sample*.

# %%
DP = SCRATCH / "deck_player.parquet"
deck_player = (obs.withColumn("nm", 1 - F.col("deck_mirror"))
               .groupBy("season_id", "deck_id", "player_tag")
               .agg(F.count("*").alias("n_all"), F.sum("nm").alias("n_nm"), F.sum(F.col("nm") * F.col("win")).alias("w_nm"),
                    *[F.sum(F.col("nm") * F.col("win") * (F.col("margin") == k).cast("int")).alias(f"w{k}") for k in (1, 2, 3)]))
if not done(DP):
    deck_player.write.mode("overwrite").parquet(str(DP))
deck_player = spark.read.parquet(str(DP))
deck_player_all = (deck_player.groupBy("deck_id", "player_tag")
                   .agg(*[F.sum(c).alias(c) for c in ["n_all", "n_nm", "w_nm", "w1", "w2", "w3"]])
                   .withColumn("season_id", F.lit("ALL")))

def deck_level(dp):
    q = F.col("n_nm") >= MIN_PLAYER_GAMES
    return (dp.groupBy("season_id", "deck_id").agg(
        F.sum("n_all").alias("obs"), F.sum("n_nm").alias("nm_matches"), F.sum("w_nm").alias("nm_wins"),
        F.sum("w1").alias("w1"), F.sum("w2").alias("w2"), F.sum("w3").alias("w3"),
        F.count("*").alias("distinct_players"), F.sum(q.cast("int")).alias("qualifying_players"),
        F.avg(F.when(q, F.col("w_nm") / F.col("n_nm"))).alias("wr_player"),
        F.max("n_all").alias("top_player_obs"))
        .filter(F.col("obs") >= LEADERBOARD_MIN_OBS))

dl = to_pd(deck_level(deck_player).unionByName(deck_level(deck_player_all.select(deck_player.columns)))
           .join(family_map, "deck_id", "left"))
dl["n_battles"] = dl.season_id.map(N_BATTLES)
dl["pick_rate"] = dl.obs / (2 * dl.n_battles)
dl["mirror_share"] = 1 - dl.nm_matches / dl.obs
dl["wr_match"] = dl.nm_wins / dl.nm_matches
dl["specialist_gap"] = dl.wr_match - dl.wr_player
dl["top_player_share"] = dl.top_player_obs / dl.obs
dl = add_crowns(apply_eb(dl, "deck"))
dl["card_ids"] = dl.deck_id.str.split("-").map(lambda x: [int(v) for v in x])
dl["card_names"] = dl.card_ids.map(lambda ids: [CARD_NAME.get(i, str(i)) for i in ids])

def classify(g):
    g = g.copy()
    elig = (g.nm_matches >= CLASS_MIN_MATCHES) & (g.qualifying_players >= CLASS_MIN_PLAYERS)
    e = g[elig]
    med_p, p75_p, q75_wr = e.distinct_players.median(), e.distinct_players.quantile(0.75), e.wr_eb.quantile(0.75)
    cls = np.where(~elig, "Insufficient sample", "Unclassified").astype(object)
    spec = elig & (g.specialist_gap > SPEC_GAP) & (g.distinct_players <= med_p)
    broad = elig & (g.wr_eb >= q75_wr) & (g.wr_player >= 0.5) & (g.distinct_players >= p75_p) & (g.specialist_gap.abs() <= SPEC_GAP)
    cls[broad] = "Broad Meta Staple"
    cls[spec] = "Specialist-Driven"
    g["deck_class"] = cls
    return g
dl = pd.concat([classify(g) for _, g in dl.groupby("season_id")], ignore_index=True)
dl["pick_rank"] = dl.groupby("season_id").pick_rate.rank(ascending=False, method="min")
dl["eb_rank"] = dl.groupby("season_id").wr_eb.rank(ascending=False, method="min")
dl["raw_wr_rank"] = dl.groupby("season_id").wr_match.rank(ascending=False, method="min")
save_parquet_pd(dl.drop(columns=["w1", "w2", "w3"]), "deck_leaderboard.parquet")
print(dl.groupby(["season_id", "deck_class"]).size().unstack(fill_value=0))

# %% [markdown]
# ### 5c. Family metrics

# %%
fo = (obs.join(family_map, "deck_id")
      .join(family_map.withColumnRenamed("deck_id", "opp_deck_id").withColumnRenamed("family_id", "opp_family_id"), "opp_deck_id")
      .withColumn("nm", (F.col("family_id") != F.col("opp_family_id")).cast("int")))
fam_player = fo.groupBy("season_id", "family_id", "player_tag").agg(
    F.count("*").alias("n_all"), F.sum("nm").alias("n_nm"), F.sum(F.col("nm") * F.col("win")).alias("w_nm"),
    *[F.sum(F.col("nm") * F.col("win") * (F.col("margin") == k).cast("int")).alias(f"w{k}") for k in (1, 2, 3)])
FP = SCRATCH / "family_player.parquet"
if not done(FP):
    fam_player.write.mode("overwrite").parquet(str(FP))
fam_player = spark.read.parquet(str(FP))
fam_variants = fo.groupBy("season_id", "family_id").agg(F.countDistinct("deck_id").alias("variants_seen"))
fam_variants_all = fo.groupBy("family_id").agg(F.countDistinct("deck_id").alias("variants_seen")).withColumn("season_id", F.lit("ALL"))
fam_player_all = (fam_player.groupBy("family_id", "player_tag")
                  .agg(*[F.sum(c).alias(c) for c in ["n_all", "n_nm", "w_nm", "w1", "w2", "w3"]])
                  .withColumn("season_id", F.lit("ALL")))
def fam_level(fp):
    q = F.col("n_nm") >= MIN_PLAYER_GAMES
    return fp.groupBy("season_id", "family_id").agg(
        F.sum("n_all").alias("obs"), F.sum("n_nm").alias("nm_matches"), F.sum("w_nm").alias("nm_wins"),
        F.sum("w1").alias("w1"), F.sum("w2").alias("w2"), F.sum("w3").alias("w3"),
        F.count("*").alias("distinct_players"), F.sum(q.cast("int")).alias("qualifying_players"),
        F.avg(F.when(q, F.col("w_nm") / F.col("n_nm"))).alias("wr_player")).filter(F.col("obs") >= LEADERBOARD_MIN_OBS)
fs = to_pd(fam_level(fam_player).unionByName(fam_level(fam_player_all.select(fam_player.columns)))
           .join(fam_variants.unionByName(fam_variants_all.select(fam_variants.columns)), ["season_id", "family_id"], "left"))
fs["n_battles"] = fs.season_id.map(N_BATTLES)
fs["pick_rate"] = fs.obs / (2 * fs.n_battles)
fs["wr_match"] = fs.nm_wins / fs.nm_matches
fs["specialist_gap"] = fs.wr_match - fs.wr_player
fs = add_crowns(apply_eb(fs, "family"))
fs["head_card_ids"] = fs.family_id.str.split("-").map(lambda x: [int(v) for v in x])
fs["head_card_names"] = fs.head_card_ids.map(lambda ids: [CARD_NAME.get(i, str(i)) for i in ids])
# short label: the head's 3 most distinctive cards (lowest overall card pick rate across all seasons)
card_pr_all = cm[cm.season_id == "ALL"].set_index("card_id").pick_rate
fs["family_label"] = fs.head_card_ids.map(
    lambda ids: " · ".join(CARD_NAME.get(i, str(i)) for i in sorted(ids, key=lambda i: card_pr_all.get(i, 0))[:3]))
fs["pick_rank"] = fs.groupby("season_id").pick_rate.rank(ascending=False, method="min")
save_parquet_pd(fs.drop(columns=["w1", "w2", "w3"]), "family_seasonal_summary.parquet")
fs[fs.season_id == "ALL"].nlargest(12, "pick_rate")[
    ["family_label", "variants_seen", "pick_rate", "wr_match", "wr_eb", "distinct_players", "specialist_gap"]]

# %%
# Gate D assertions
eb_df = pd.DataFrame(EB_PRIORS)
display_cols = ["entity", "scope", "mu0", "M", "alpha", "beta", "fit"]
print(eb_df[display_cols].to_string(index=False))
for name, d in [("card", cm), ("deck", dl), ("family", fs)]:
    d = d[d.nm_matches > 0]
    raw_dev = (d.wr_match if "wr_match" in d else d.wr_nonmirror) - d.eb_prior_mu
    eb_dev = d.wr_eb - d.eb_prior_mu
    assert (eb_dev.abs() <= raw_dev.abs() + 1e-12).all(), f"EB must shrink toward the prior mean ({name})"
    s = d[d.nm_wins > 0][["win_1crown_pct", "win_2crown_pct", "win_3crown_pct"]].sum(axis=1)
    assert np.allclose(s, 1.0), f"crown margins must sum to 100% ({name})"
# non-mirror exclusion: no non-mirror deck match is against the identical deck; no card nm match has the card on both sides
assert obs.filter("deck_mirror = 1 AND deck_id != opp_deck_id").count() == 0
assert card_obs.filter("nm = 1 AND array_contains(opp_cards, card_id)").limit(1).count() == 0
assert (dl.nm_matches <= dl.obs).all() and (cm.nm_matches <= cm.obs).all()
assert dl.specialist_gap.notna().any() and dl.wr_player.dropna().between(0, 1).all()
print("Gate D passed: mirrors excluded; EB shrinks toward mu0; crown shares sum to 100%; specialist gaps computed")

# %% [markdown]
# ## 6. Cross-season temporal dynamics (Gate E)
#
# Spearman ρ of card, deck and family pick rates and non-mirror win rates across season pairs.
# Entities enter a pair only if they meet a floor in *both* seasons:
# cards ≥ 1,000 observations; decks and families ≥ 200 non-mirror matches.
# The table also holds non-adjacent pairs (full correlation matrices) and per-card rank shifts.
#
# **Card trajectory categories** (evaluated in this order; first match wins):
# * *Permanent Staple*: pick-rate rank in the top 15% of cards in all 5 seasons.
# * *Balance Casualty*: at some season boundary, pick rate falls ≥ 35% relative and ≥ 1 pp absolute,
#   **or** non-mirror WR falls ≥ 2 pp while pick rate is ≥ 1%.
# * *Emergent Riser*: pick rate in S05 ≥ 2× S01, and the S05 pick-rate percentile ≥ 70%.
# * *Volatile / Seasonal Fad*: one season's pick rate ≥ 1.5× the median of the other four, with that
#   peak ≥ 2 pp, and the card never in the top 15% in two or more seasons.
# * Otherwise *Stable / Other*.
#
# Season resets (when Supercell ships balance changes) fall on the season boundaries. The prompt
# names Phoenix (26000087), Monk (26000077) and Rage (28000002) as patch-affected cards; they are
# tracked weekly. Their exact patch dates and contents were not checked against official patch
# notes here, so the trajectories are described, not attributed to a specific change.

# %%
SIDS = list(SEASONS)
recs = []
def spearman_block(df, entity, key, floor_col, floor):
    for metric in ["pick_rate", "wr_nonmirror" if entity == "card" else "wr_match", "wr_eb"]:
        for i, a in enumerate(SIDS):
            for b in SIDS[i + 1:]:
                A = df[(df.season_id == a) & (df[floor_col] >= floor)].set_index(key)[metric]
                B = df[(df.season_id == b) & (df[floor_col] >= floor)].set_index(key)[metric]
                common = A.index.intersection(B.index)
                if len(common) >= 5:
                    rho, p = stats.spearmanr(A[common], B[common])
                else:
                    rho, p = np.nan, np.nan
                recs.append({"record_type": "spearman", "entity": entity, "metric": metric.replace("wr_nonmirror", "wr_match"),
                             "season_a": a, "season_b": b, "adjacent": SIDS.index(b) - i == 1,
                             "n_common": len(common), "rho": rho, "p_value": p})
spearman_block(cm, "card", "card_id", "obs", 1000)
spearman_block(dl, "deck", "deck_id", "nm_matches", 200)
spearman_block(fs, "family", "family_id", "nm_matches", 200)
corr = pd.DataFrame(recs)

# per-card rank shifts across adjacent seasons
cs = cm[cm.season_id != "ALL"]
pr = cs.pivot(index="card_id", columns="season_id", values="pick_rate").reindex(columns=SIDS).fillna(0)
wr = cs.pivot(index="card_id", columns="season_id", values="wr_nonmirror").reindex(columns=SIDS)
rk = pr.rank(ascending=False, method="min")
shift_recs = []
for a, b in zip(SIDS[:-1], SIDS[1:]):
    for cid in pr.index:
        shift_recs.append({"record_type": "card_rank_shift", "entity": "card", "metric": "pick_rate",
                           "season_a": a, "season_b": b, "adjacent": True, "entity_id": str(cid),
                           "entity_name": CARD_NAME.get(cid, str(cid)),
                           "value_a": pr.at[cid, a], "value_b": pr.at[cid, b],
                           "rank_a": rk.at[cid, a], "rank_b": rk.at[cid, b], "rank_shift": rk.at[cid, a] - rk.at[cid, b],
                           "wr_a": wr.at[cid, a], "wr_b": wr.at[cid, b]})
stability = pd.concat([corr, pd.DataFrame(shift_recs)], ignore_index=True)
save_parquet_pd(stability, "cross_season_stability.parquet")
adj = corr[corr.adjacent].pivot_table(index=["entity", "metric"], columns=["season_a", "season_b"], values="rho")
adj.round(3)

# %%
n_cards = len(pr)
top15 = rk <= math.ceil(0.15 * n_cards)
pctile = pr.rank(pct=True)
def trajectory(cid):
    p, wv = pr.loc[cid], wr.loc[cid]
    if top15.loc[cid].all():
        return "Permanent Staple"
    for a, b in zip(SIDS[:-1], SIDS[1:]):
        if (p[a] - p[b] >= 0.01 and p[b] <= 0.65 * p[a]) or (p[a] >= 0.01 and p[b] >= 0.01 and wv[a] - wv[b] >= 0.02):
            return "Balance Casualty"
    if p["S01"] > 0 and p["S05"] >= 2 * p["S01"] and pctile.at[cid, "S05"] >= 0.70:
        return "Emergent Riser"
    for s in SIDS:
        others = p.drop(s)
        if p[s] >= 1.5 * others.median() and p[s] >= 0.02 and top15.loc[cid].sum() < 2:
            return "Volatile / Seasonal Fad"
    return "Stable / Other"
traj = pd.DataFrame({"card_id": pr.index})
traj["card_name"] = traj.card_id.map(lambda c: CARD_NAME.get(c, str(c)))
traj["trajectory"] = traj.card_id.map(trajectory)
for s in SIDS:
    traj[f"pick_{s}"] = traj.card_id.map(pr[s]); traj[f"wr_{s}"] = traj.card_id.map(wr[s])
save_parquet_pd(traj, "card_trajectories.parquet")
for k, g in traj.groupby("trajectory"):
    print(f"{k} ({len(g)}): " + ", ".join(g.sort_values(f"pick_S05", ascending=False).card_name.head(14)))

# %%
focus = card_week[card_week.card_id.isin(FOCUS_CARDS)]
focus = focus.groupby(["card_id", "week"], as_index=False)[["obs", "nm_matches", "nm_wins"]].sum().merge(
    week_n.groupby("week", as_index=False).n_battles.sum(), on="week")
focus["pick_rate"] = focus.obs / (2 * focus.n_battles)
focus["wr_nonmirror"] = focus.nm_wins / focus.nm_matches
focus["card_name"] = focus.card_id.map(FOCUS_CARDS)
focus = focus[focus.n_battles >= 1000].sort_values(["card_id", "week"])   # drop the thin collection ramp-up / tail weeks
traj[traj.card_id.isin(FOCUS_CARDS)][["card_name", "trajectory"] + [f"pick_{s}" for s in SIDS] + [f"wr_{s}" for s in SIDS]]

# %% [markdown]
# ## 7. Figures (300 DPI, `./outputs/`)

# %%
# Figure 1 - seasonal trophy profiling & threshold shift (small multiples, one per season)
hist = to_pd(ladder_obs.filter("trophies >= 5000").withColumn("bin", (F.floor(F.col("trophies") / 25) * 25).cast("int"))
             .groupBy("season_id", "bin").count()).sort_values(["season_id", "bin"])
fig, axes = plt.subplots(len(SIDS), 1, figsize=(7.2, 7.6), sharex=True)
for ax, s in zip(axes, SIDS):
    h = hist[hist.season_id == s]
    dens = h["count"] / h["count"].sum()
    ax.fill_between(h.bin + 12.5, dens, step="mid", color=RAMP["250"], linewidth=0)
    ax.plot(h.bin + 12.5, dens, drawstyle="steps-mid", color=C1, linewidth=1.2)
    r = thr.set_index("season_id").loc[s]
    for q, ls in [("p50", ":"), ("p90", "--")]:
        ax.axvline(r[q], color=MUTED, linestyle=ls, linewidth=1)
    ax.axvline(r["tau"], color=C2, linewidth=2)
    ax.text(r["tau"] - 15, ax.get_ylim()[1] * 0.85, f"τ = {int(r['tau']):,}", color=INK, ha="right", fontsize=8.5, fontweight="semibold")
    ax.text(0.005, 0.80, f"{s} · {SEASON_LABEL[s]}", transform=ax.transAxes, fontsize=9, color=INK)
    ax.text(0.005, 0.58, f"median {int(r['p50']):,} · at cap {r['share_at_cap']:.0%}", transform=ax.transAxes, fontsize=8, color=INK2)
    ax.set_yticks([]); ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
axes[0].set_title("Figure 1. Ladder starting-trophy distribution per season and the P75 threshold τ_s")
axes[-1].set_xlabel("Starting trophies (Ladder + Ladder Gold Rush, both players; 25-trophy bins, ≥ 5,000 shown)")
axes[-1].xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
fig.text(0.01, -0.01, "Orange line: τ_s (P75). Dotted: P50. Dashed: P90. Distributions are shares of player-observations within each season.",
         fontsize=7.5, color=INK2)
fig.savefig(OUT / "fig1_seasonal_trophy_thresholds.png"); plt.show()

# %%
# Figure 2 - card pick-vs-win quadrant chart (all five seasons pooled)
ca = cm[(cm.season_id == "ALL") & (cm.obs >= 5000)].copy()
x_split, y_split = ca.pick_rate.median(), 0.5
fig, ax = plt.subplots(figsize=(7.6, 5.6))
is_focus = ca.card_id.isin(FOCUS_CARDS)
ax.scatter(ca.pick_rate[~is_focus], ca.wr_eb[~is_focus], s=26, color=C1, edgecolor=SURF, linewidth=1, alpha=0.9, zorder=3, label="Card")
ax.scatter(ca.pick_rate[is_focus], ca.wr_eb[is_focus], s=46, color=C2, edgecolor=SURF, linewidth=1.2, zorder=4, label="Phoenix / Monk / Rage")
ax.set_xscale("log")
ax.axvline(x_split, color=MUTED, linewidth=1); ax.axhline(y_split, color=MUTED, linewidth=1)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0)); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
xl, yl = ax.get_xlim(), ax.get_ylim()
for txt, xx, yy, ha, va in [("Meta Staples", xl[1], yl[1], "right", "top"), ("Overrated Traps", xl[1], yl[0], "right", "bottom"),
                            ("Niche / Sleepers", xl[0], yl[1], "left", "top"), ("Underpowered", xl[0], yl[0], "left", "bottom")]:
    ax.text(xx, yy, txt, ha=ha, va=va, fontsize=9, color=INK2, fontweight="semibold", zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", facecolor=SURF, edgecolor="none", alpha=0.9))
lab = pd.concat([ca.nlargest(8, "pick_rate"), ca.nlargest(5, "wr_eb"), ca.nsmallest(5, "wr_eb"), ca[is_focus]]).drop_duplicates("card_id")
anns = [ax.annotate(r.card_name, (r.pick_rate, r.wr_eb), xytext=(5, 0), textcoords="offset points", fontsize=7.2,
                    color=INK, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3))
        for _, r in lab.iterrows()]
declutter(ax, anns)
ax.set_xlabel("Pick rate (share of player-observations containing the card, log scale)\n"
              f"Quadrant split: median pick rate ({x_split:.1%}) and 50% win rate. Cards with ≥ 5,000 observations.")
ax.set_ylabel("Non-mirror win rate (EB-smoothed)")
ax.set_title("Figure 2. Card pick rate vs. non-mirror win rate, high-trophy battles, Nov 2022 – Mar 2023")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.savefig(OUT / "fig2_card_pick_vs_win_quadrants.png"); plt.show()

# %%
# Figure 3 - deck breadth & specialist gap
da = dl[(dl.season_id == "ALL") & (dl.deck_class != "Insufficient sample")].copy()
cls_color = {"Broad Meta Staple": C1, "Specialist-Driven": C2, "Unclassified": "#b9b8b2"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
for k in ["Unclassified", "Broad Meta Staple", "Specialist-Driven"]:
    d = da[da.deck_class == k]
    a1.scatter(d.wr_player, d.wr_match, s=10 + 30 * np.sqrt(d.distinct_players / da.distinct_players.max()),
               color=cls_color[k], edgecolor=SURF, linewidth=0.6, alpha=0.85, label=f"{k} ({len(d)})", zorder=3 if k != "Unclassified" else 2)
    a2.scatter(d.distinct_players, d.specialist_gap, s=14, color=cls_color[k], edgecolor=SURF, linewidth=0.6, alpha=0.85,
               zorder=3 if k != "Unclassified" else 2)
lo, hi_ = np.nanpercentile(np.r_[da.wr_player, da.wr_match], [0.5, 99.5])
a1.plot([lo, hi_], [lo, hi_], color=MUTED, linewidth=1, linestyle="--")
a1.set_xlim(lo, hi_); a1.set_ylim(lo, hi_)
a1.set_xlabel("Player-weighted WR (mean over players with ≥ 5 games)"); a1.set_ylabel("Match-weighted non-mirror WR")
a1.set_title("A. Match- vs. player-weighted win rate")
for a in (a1,):
    a.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0)); a.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
a2.set_xscale("log"); a2.axhline(SPEC_GAP, color=C2, linewidth=1, linestyle="--"); a2.axhline(0, color=MUTED, linewidth=1)
a2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
a2.set_xlabel("Distinct players |P_D| (log scale)"); a2.set_ylabel("Specialist gap Δ = WR_match − WR_player")
a2.set_title("B. Specialist gap vs. breadth of player pool")
a2.text(a2.get_xlim()[1], SPEC_GAP, " +3 pp", va="bottom", ha="right", fontsize=7.5, color=INK2)
fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Figure 3. Deck breadth and specialist gap, all seasons pooled "
             f"(decks with ≥ {CLASS_MIN_MATCHES} non-mirror matches and ≥ {CLASS_MIN_PLAYERS} qualifying players)",
             x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
fig.tight_layout(); fig.savefig(OUT / "fig3_deck_breadth_specialist_gap.png"); plt.show()

# %%
# Figure 4 - crown-margin profiles across the top deck families (ordinal ramp: 1 → 3 crowns)
tf = fs[fs.season_id == "ALL"].nlargest(15, "pick_rate").iloc[::-1]
fig, ax = plt.subplots(figsize=(8.2, 5.8))
left = np.zeros(len(tf))
for k, col in zip((1, 2, 3), (RAMP["250"], RAMP["400"], RAMP["600"])):
    v = tf[f"win_{k}crown_pct"].values
    ax.barh(range(len(tf)), v, left=left, color=col, edgecolor=SURF, linewidth=2, height=0.72,
            label=f"Won by {k} crown{'s' if k > 1 else ''}")
    for i, (l, vv) in enumerate(zip(left, v)):
        if vv >= 0.08:
            ax.text(l + vv / 2, i, f"{vv:.0%}", ha="center", va="center", fontsize=7, color=INK if k == 1 else "#ffffff")
    left += v
ax.set_yticks(range(len(tf)), [f"{l}  ({p:.1%} pick, {w:.1%} WR)" for l, p, w in zip(tf.family_label, tf.pick_rate, tf.wr_eb)], fontsize=7.6)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0)); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.set_xlabel("Share of non-mirror wins by crown margin (|winner crowns − loser crowns|)")
ax.set_title("Figure 4. Crown-margin profile of wins, 15 most-played deck families (all seasons)")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16))
fig.text(0.01, -0.10, "Margin 1 covers 1–0, 2–1 and 3–2 results; margin 3 is a 3–0 shutout.\n"
         "Family label = the head deck's three least-common cards. Pick rate and EB-smoothed non-mirror WR at family level.",
         fontsize=7.5, color=INK2)
fig.savefig(OUT / "fig4_crown_margin_profiles.png"); plt.show()

# %%
# Figure 5 - weekly trajectories of Phoenix, Monk and Rage (two aligned panels, no dual axis)
import matplotlib.dates as mdates
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
end_labels, end_labels2 = [], []
for (cid, name), col in zip(FOCUS_CARDS.items(), (C1, C2, C3)):
    d = focus[focus.card_id == cid]
    a1.plot(d.week, d.pick_rate, color=col, marker="o", markersize=3.5, label=name)
    a2.plot(d.week, d.wr_nonmirror, color=col, marker="o", markersize=3.5, label=name)
    end_labels.append(a1.annotate(name, (d.week.iloc[-1], d.pick_rate.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK))
    end_labels2.append(a2.annotate(name, (d.week.iloc[-1], d.wr_nonmirror.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK))
a2.axhline(0.5, color=MUTED, linewidth=1)
for s, (_, d0, _) in SEASONS.items():
    for a in (a1, a2):
        a.axvline(pd.Timestamp(d0), color=MUTED, linewidth=0.8, linestyle=":")
    a1.text(pd.Timestamp(d0), 1.0, f" {s}", transform=a1.get_xaxis_transform(), fontsize=7.5, color=INK2, va="top")
a1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0)); a2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
a1.set_ylabel("Weekly pick rate"); a2.set_ylabel("Weekly non-mirror WR")
a2.xaxis.set_major_locator(mdates.MonthLocator()); a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
declutter(a1, end_labels); declutter(a2, end_labels2)
a1.set_title("Figure 5. Weekly pick and non-mirror win rate of Phoenix, Monk and Rage (high-trophy battles)")
a1.legend(loc="upper right", ncol=3)
a2.set_xlabel("Week (Monday start, UTC). Dotted lines: season boundaries.")
fig.savefig(OUT / "fig5_balance_patch_trajectories.png"); plt.show()

# %% [markdown]
# ## 8. Deliverable audit (Gate F)

# %%
from PIL import Image
REQUIRED_TABLES = {
    "seasonal_trophy_thresholds.parquet": ["season_id", "p50", "p75", "p80", "p90", "tau"],
    "card_seasonal_metrics.parquet": ["season_id", "card_id", "card_name", "pick_rate", "wr_nonmirror", "wr_eb"],
    "deck_leaderboard.parquet": ["season_id", "deck_id", "card_ids", "family_id", "obs", "distinct_players", "wr_match",
                                 "wr_player", "wr_eb", "specialist_gap", "win_1crown_pct", "win_2crown_pct", "win_3crown_pct", "deck_class"],
    "family_seasonal_summary.parquet": ["season_id", "family_id", "pick_rate", "wr_match", "wr_eb", "win_1crown_pct"],
    "cross_season_stability.parquet": ["record_type", "entity", "metric", "season_a", "season_b", "rho", "rank_shift"],
}
for name, cols in REQUIRED_TABLES.items():
    d = pd.read_parquet(OUT / name)
    missing = set(cols) - set(d.columns)
    assert not missing and len(d) > 0, (name, missing)
    seasons = set(d.season_id) if "season_id" in d else set(d.season_a) | set(d.season_b)
    assert set(SIDS) <= seasons, (name, seasons)
    print(f"✓ {name:38s} {len(d):>9,} rows")
FIGS = ["fig1_seasonal_trophy_thresholds.png", "fig2_card_pick_vs_win_quadrants.png", "fig3_deck_breadth_specialist_gap.png",
        "fig4_crown_margin_profiles.png", "fig5_balance_patch_trajectories.png"]
for f in FIGS:
    im = Image.open(OUT / f)
    dpi = im.info.get("dpi", (0, 0))[0]
    assert round(dpi) >= 299, (f, dpi)
    print(f"✓ {f:38s} {im.size[0]}×{im.size[1]} px @ {round(dpi)} dpi")
assert spark.read.parquet(str(STAGED)).select("season_id").distinct().count() == 5
print(f"Gate F passed. Total runtime {(time.time() - T0) / 60:.1f} min")

# %% [markdown]
# ## 9. Findings summary (computed)

# %%
def pct(x): return f"{x:.1%}"
r = thr.set_index("season_id")
print("Trophy thresholds τ_s (P75):", ", ".join(f"{s} {int(r.at[s,'tau']):,}" for s in SIDS))
print("Share of ladder battles where both players ≥ 7,000 (static rule):",
      ", ".join(f"{s} {pct(r.at[s,'share_ladder_battles_qualifying_static7000'])}" for s in SIDS))
print("Share qualifying under τ_s:", ", ".join(f"{s} {pct(r.at[s,'share_ladder_battles_qualifying_tau'])}" for s in SIDS))
print("High-trophy (non-tie) battles:", ", ".join(f"{s} {N_BATTLES[s]:,}" for s in SIDS), "| total", f"{N_BATTLES['ALL']:,}")
c = cm[cm.season_id == "ALL"]
rho_pw = stats.spearmanr(c.pick_rate, c.wr_eb)
print(f"Cards (pooled): Spearman(pick rate, EB WR) = {rho_pw.statistic:.3f} (n={len(c)})")
d = dl[(dl.season_id == "ALL") & (dl.nm_matches >= CLASS_MIN_MATCHES)]
rho_d = stats.spearmanr(d.pick_rate, d.wr_eb)
print(f"Decks ≥{CLASS_MIN_MATCHES} matches (pooled): Spearman(pick rate, EB WR) = {rho_d.statistic:.3f} (n={len(d)})")
top = dl[dl.season_id == "ALL"].nsmallest(20, "raw_wr_rank")
print(f"Top-20 decks by raw WR (≥{LEADERBOARD_MIN_OBS} obs): median matches {top.nm_matches.median():.0f}; "
      f"median EB rank {top.eb_rank.median():.0f}")
for e in ["card", "deck", "family"]:
    x = corr[(corr.entity == e) & corr.adjacent]
    print(f"{e:6s} adjacent-season ρ: pick " + ", ".join(f"{v:.2f}" for v in x[x.metric == 'pick_rate'].rho)
          + " | WR " + ", ".join(f"{v:.2f}" for v in x[x.metric == 'wr_match'].rho))
print(dl[dl.season_id == "ALL"].deck_class.value_counts().to_string())
score = to_pd(obs.filter("win = 1").groupBy("crowns", "opp_crowns").count())
score["share"] = score["count"] / score["count"].sum()
print("Final score of wins (winner–loser crowns), all seasons:\n" + score.sort_values(["crowns", "opp_crowns"]).to_string(index=False))

# %% [markdown]
# ### Interpretation
#
# All numbers below come from the executed cells above. They are **descriptive** statements about
# high-trophy Trophy Road battles from 8 Nov 2022 to 3 Apr 2023. They do not establish card strength
# or player skill.
#
# **Data and sample.**
# * 149 files gave 193.9 M raw rows. None failed validation.
# * 2.77 M duplicate lines were removed. All of them come from the two boundary-day files stored in two
#   season folders (`20230206`, `20230306`).
# * No swapped-perspective duplicates exist, so the canonical-key dedup is a safeguard that turned out
#   not to be needed.
# * 191.1 M clean battles remain. Two mode IDs that are not in the taxonomy (72000201, 72000328;
#   1–2 M battles per season) are not on the collection list and are excluded. Their exact identity was
#   not verified.
#
# **Seasonal thresholds matter.**
# * Ladder trophies inflate steadily: median 6,647 → 7,264, and the share of players at the 7,500 cap
#   grows from 0.6% → 16.3%.
# * A fixed "both players ≥ 7,000" rule would keep 14.7% of ladder battles in Nov 2022 but 99.5% in
#   Mar 2023. It would compare a top slice with almost everyone.
# * τ_s = P75 (6,798 / 7,087 / 7,242 / 7,305 / 7,384) keeps a stable 20–22% each season.
# * The Path of Legends trophy field holds league points (P50 = 30), so PoL battles never meet a
#   ladder τ_s. The high-trophy sample is in practice Ladder, Ladder Gold Rush and Showdown Ladder:
#   12.28 M non-tie battles. The 1,695 ties were excluded.
#
# **Popularity is only weakly tied to winning.**
# * Across 109 cards, Spearman ρ(pick rate, EB win rate) = 0.11. Across 11,253 decks with ≥ 200
#   matches, ρ = 0.13.
# * The most-picked card, The Log (35.8% of observations), has a 49.5% non-mirror win rate.
#   Ice Spirit, Hog Rider, Skeletons and Valkyrie (13–14% pick) all sit below 49%: the
#   *Overrated Traps* quadrant of Figure 2.
# * Phoenix (20.8% pick, 52.4%) and Barbarian Barrel (18.9%, 52.7%) are the high-pick, high-win
#   *Meta Staples*.
# * Mirror matches are frequent for staples: 36% of The Log's observations face another Log. Without
#   the non-mirror correction, these win rates would be pulled toward 50%.
#
# **Small samples dominate raw deck leaderboards.**
# * The 20 decks with the highest raw win rate (≥ 50 observations) have a median of only 62 matches.
#   Their median rank after Empirical-Bayes smoothing is 90.
# * The fitted prior strength M falls from ~210–270 pseudo-matches (Nov–Dec) to ~27–34 (Feb–Mar).
#   Win rates of decks with ≥ 100 matches spread out more widely in later seasons.
# * The deck prior mean (0.503–0.517) is above 50%. This is a selection effect: decks played often
#   enough to reach 100 matches tend to beat the long tail of rare decks.
#
# **Breadth vs. specialisation.**
# * Among 4,997 pooled decks with enough data, 310 are *Specialist-Driven* (Δ_spec > +3 pp with a
#   small player pool, typically 10–60 pilots) and 258 are *Broad Meta Staples*.
# * Figure 3B shows positive specialist gaps almost only in small player pools. For decks with
#   hundreds to thousands of pilots, Δ_spec sits near 0 or slightly below.
# * Caveat: WR_player averages only players with ≥ 5 games, while WR_match includes every match. The
#   gap therefore also reflects how occasional pilots perform, not only how specialists perform.
#
# **Deck families.**
# * 1.07 M distinct decks collapse into 740 K families. The top 100 families cover 40% of observations.
# * Transitive clustering on shared 7-card cores would have merged 80.5% of seed-deck volume into a
#   single component. That supports the non-transitive head assignment used here: largest family = 2.0%.
#
# **Crown margins.**
# * Most wins are narrow. Across all seasons, 1–0 is by far the most common final score (two-thirds
#   of wins in S03), and margin-1 wins make up 68–91% of wins for the top-15 families.
# * The Golem / Elixir Collector beatdown family has the largest share of 2- and 3-crown-margin wins
#   (32%). This fits the beatdown label, but it is descriptive only.
#
# **Cross-season stability.**
# * Card pick rates are very persistent (adjacent-season ρ 0.81–0.98), and card win rates mostly so
#   (0.82–0.92). The exception is Nov → Dec (ρ = 0.58), which is the least stable transition for
#   every entity and metric.
# * Exact decks turn over faster (pick ρ 0.66–0.76; WR ρ 0.39–0.68). Families sit in between,
#   consistent with archetypes persisting while individual card slots get swapped.
#
# **Phoenix, Monk, Rage (Figure 5).**
# * Phoenix's weekly pick rate peaked at ~45% in mid-November. It began falling in the last week of
#   S01 (week of 28 Nov), before the S02 boundary, and settled at 14–18% from December on. Its
#   non-mirror win rate stayed above 50% for most weeks, so the trajectory rule classifies it as a
#   *Permanent Staple*.
# * Monk fell from 14.8% to 2.3% pick and from 51.9% to 45.1% win rate across the S01 → S02 boundary
#   (*Balance Casualty*). It partly recovered in win rate from February.
# * Rage drifted from 8.5% to 5–6% pick at a 48–50% win rate (*Stable / Other*).
# * These timings line up with season boundaries and a mid-season change. The battle data alone
#   cannot attribute them to specific balance changes, and patch contents were not cross-checked here.
# * The last plotted week (3 Apr 2023) contains a single partial day and should not be read as a trend.
#
# **Deviations from the original specification (all documented in the cells above).**
# * Spark runs `local[4]` with an 8 GB heap, matching the container's cgroup limits, instead of
#   `local[*]` with 128 GB.
# * Seasons are assigned by battle date, not folder, because two days appear in two season folders.
# * Deck families use a non-transitive, volume-ordered head assignment instead of connected components.
# * Ties (~0.01%) are excluded.
# * Patch attributions in the original brief are treated as unverified.
