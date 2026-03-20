from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent


DATA_PATH = ROOT / "data" / "aggregated_stats.csv"
OUT_DIR   = ROOT / "output"
OUT_DATASETS = OUT_DIR / "output_datasets"
TREE_DIR  = OUT_DIR / "decision_trees"
FIG_DIR   = OUT_DIR / "figures"

# ── Column names ─────────────────────────────────────────────────────────────
TARGET = "dl.throughput.value"

META_COLS = [
    "test",
    "limitation",
    "throughputlim",
    "latencylim",
    "packetlosslim",
    "dl.throughput.unit",
    "dl.latency.unit",
    "dl.retransmission.unit",
]

THROUGHPUT_LIMS = [
    "1000kbit",
    "2000kbit",
    "5000kbit",
    "10000kbit",
    "20000kbit",
    "50000kbit",
]

# ── Preprocessing ─────────────────────────────────────────────────────────────
MISSING_MAX = 0.01   # max fraction of missing values allowed per column
VIF_MAX     = 20.0   # max Variance Inflation Factor before pruning

# ── GMM / BIC clustering ──────────────────────────────────────────────────────
EVAL_WINDOW    = 40   # max number of clusters to evaluate
BIC_ITERATIONS = 50   # Monte Carlo iterations for BIC curve estimation

# ── Anomaly detection ─────────────────────────────────────────────────────────
MAD_THRESHOLD = 3.0   # threshold multiplier for MAD-based outlier detection

# ── Decision tree ─────────────────────────────────────────────────────────────
DT_MAX_DEPTH    = 10
DT_MIN_SAMPLES  = 1
DT_CROSS_VAL    = 6

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
REGRESSOR    = "XGB"