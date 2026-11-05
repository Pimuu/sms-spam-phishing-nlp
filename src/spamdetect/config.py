"""Central configuration: paths, random seeds, and experiment settings.

Everything that influences a reported number lives here so that a single
file documents the experimental protocol.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the repository root, resolved from this file)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "SMSSpamCollection.tsv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Dataset provenance
# --------------------------------------------------------------------------
# The SMS Spam Collection v.1 (Almeida, Gomez Hidalgo & Yamakami, 2011),
# distributed by the UCI Machine Learning Repository under CC BY 4.0.
# The official UCI archive is tried first; a long-lived GitHub mirror of the
# identical 5,574-line file is the fallback. A copy is committed in data/raw/
# so the pipeline does not depend on the network at all.
DATASET_SOURCES = [
    # (url, format) -- "uci_zip" is the official zip; "tsv" is a plain TSV.
    ("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip", "uci_zip"),
    ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", "tsv"),
]
# Integrity checks: the canonical release has exactly these counts.
EXPECTED_N_MESSAGES = 5574
EXPECTED_N_SPAM = 747
# SHA-256 of the committed TSV (data/raw/SMSSpamCollection.tsv).
EXPECTED_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"

# --------------------------------------------------------------------------
# Experimental protocol
# --------------------------------------------------------------------------
SEED = 42                # global random seed (split, CV folds, models, bootstrap)
TEST_SIZE = 0.20         # stratified hold-out fraction
CV_FOLDS = 5             # stratified K-fold for hyper-parameter selection
CV_SCORING = "f1"        # model-selection criterion (positive class = spam)
N_BOOTSTRAP = 1000       # bootstrap resamples for test-set confidence intervals
N_PERMUTATION_REPEATS = 10  # permutation-importance repeats (hand-crafted features)
DECISION_THRESHOLD = 0.5    # probability threshold for the "spam" decision

LABEL_NAMES = {0: "ham", 1: "spam"}
