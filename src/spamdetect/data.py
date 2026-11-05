"""Dataset acquisition, integrity checking, de-duplication and splitting."""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "spam-nlp-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_dataset(dest: Path = config.RAW_DATA_PATH) -> Path:
    """Download the SMS Spam Collection to ``dest`` (TSV: ``label<TAB>text``).

    Sources in ``config.DATASET_SOURCES`` are tried in order. The official UCI
    zip contains a file named ``SMSSpamCollection`` that already has the same
    tab-separated layout, so both formats normalise to one TSV.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url, fmt in config.DATASET_SOURCES:
        try:
            payload = _fetch(url)
            if fmt == "uci_zip":
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    payload = zf.read("SMSSpamCollection")
            dest.write_bytes(payload)
            _validate_counts(_read_tsv(dest))  # raises if the file is not the canonical set
            print(f"[data] downloaded dataset from {url}")
            return dest
        except Exception as exc:  # network errors, bad archives, wrong content
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the dataset from any source:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_tsv(path: Path) -> pd.DataFrame:
    """Parse ``label<TAB>text`` lines manually.

    The messages contain unbalanced quote characters, so a hand-rolled split
    on the *first* tab is more robust than ``pd.read_csv`` quoting rules.
    """
    labels, texts = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            label, text = line.split("\t", 1)
            labels.append(label.strip().lower())
            texts.append(text)
    df = pd.DataFrame({"label_name": labels, "text": texts})
    if not set(df["label_name"]) <= {"ham", "spam"}:
        raise ValueError(f"Unexpected labels: {sorted(set(df['label_name']))}")
    df["label"] = (df["label_name"] == "spam").astype(int)
    return df


def _validate_counts(df: pd.DataFrame) -> None:
    n, n_spam = len(df), int(df["label"].sum())
    if n != config.EXPECTED_N_MESSAGES or n_spam != config.EXPECTED_N_SPAM:
        raise ValueError(
            f"Dataset has {n} messages / {n_spam} spam; expected "
            f"{config.EXPECTED_N_MESSAGES} / {config.EXPECTED_N_SPAM}."
        )


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw corpus, downloading it first if it is missing."""
    if not path.exists():
        download_dataset(path)
    digest = _sha256(path)
    if digest != config.EXPECTED_SHA256:
        # A different but valid copy (e.g. the UCI original with CRLF endings)
        # is acceptable as long as the class counts match.
        print(f"[data] note: SHA-256 {digest[:12]}... differs from the committed copy")
    df = _read_tsv(path)
    _validate_counts(df)
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate messages (after whitespace/case normalisation).

    The SMS Spam Collection contains 415 repeated messages: verbatim spam
    broadcasts and stock ham replies ("Sorry, I'll call later"). If duplicates were split across train
    and test, the test score would partly measure memorisation. We keep the
    first occurrence of each message. No message in the corpus appears with
    both labels; this is asserted below.
    """
    key = df["text"].str.lower().str.split().str.join(" ")
    conflicting = df.assign(_k=key).groupby("_k")["label"].nunique()
    assert (conflicting <= 1).all(), "a message appears with contradictory labels"
    keep = ~key.duplicated(keep="first")
    out = df.loc[keep].reset_index(drop=True)
    stats = {
        "n_raw": int(len(df)),
        "n_raw_spam": int(df["label"].sum()),
        "n_duplicates_removed": int((~keep).sum()),
        "n_duplicates_removed_spam": int(df.loc[~keep, "label"].sum()),
        "n_dedup": int(len(out)),
        "n_dedup_spam": int(out["label"].sum()),
        "n_dedup_ham": int((out["label"] == 0).sum()),
    }
    return out, stats


def split(df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    X = df["text"].to_numpy(dtype=object)
    y = df["label"].to_numpy(dtype=int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Convenience: raw load followed by de-duplication."""
    return deduplicate(load_raw())


if __name__ == "__main__":  # python -m spamdetect.data  -> (re-)download + summary
    if not config.RAW_DATA_PATH.exists():
        download_dataset()
    df, stats = load_dataset()
    print(stats)
    print(np.bincount(df["label"]))
