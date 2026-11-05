"""Feature extraction: TF-IDF, hand-crafted phishing/spam cues, and their union.

Three feature sets are compared:

``tfidf``        word uni+bi-gram TF-IDF over the cleaned text (``preprocess``).
``handcrafted``  17 interpretable statistics of the *raw* message: links,
                 phone numbers, money, urgency/reward/credential lexicons,
                 capitalisation, punctuation, length.
``combined``     horizontal concatenation of the two (``FeatureUnion``).
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from . import preprocess as pp

# --------------------------------------------------------------------------
# Cue lexicons. They are deliberately small and hand-written from the
# phishing/spam literature (see paper §2.3), not mined from this dataset, so
# the hand-crafted model is not trained on its own test vocabulary.
# --------------------------------------------------------------------------
URGENCY_WORDS = {
    "urgent", "urgently", "immediately", "now", "asap", "today", "tonight",
    "expire", "expires", "expiring", "expired", "limited", "last", "final",
    "hurry", "quick", "quickly", "deadline", "act", "important", "alert",
    "warning", "attention", "soon", "instantly",
}
REWARD_WORDS = {
    "free", "win", "winner", "won", "wins", "prize", "prizes", "cash", "claim",
    "award", "awarded", "bonus", "guaranteed", "selected", "reward", "gift",
    "voucher", "vouchers", "offer", "deal", "discount", "lottery", "jackpot",
    "congratulations", "congrats", "lucky", "chance", "credit",
}
CALL_TO_ACTION_WORDS = {
    "call", "text", "txt", "reply", "click", "send", "visit", "subscribe",
    "unsubscribe", "stop", "dial", "apply", "order", "join", "register",
    "download", "redeem", "collect",
}
CREDENTIAL_WORDS = {   # classic phishing pretexts (account takeover, payment)
    "account", "password", "bank", "verify", "verification", "login", "log",
    "security", "secure", "confirm", "pin", "update", "suspended", "locked",
    "unusual", "activity", "identity", "billing", "payment", "card", "ssn",
}

_WORD_RE = re.compile(r"[A-Za-z']+")
_LETTER_RE = re.compile(r"[A-Za-z]")
_UPPER_RE = re.compile(r"[A-Z]")
_DIGIT_RE = re.compile(r"\d")
_PUNCT_RE = re.compile(r"[^\w\s]")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")

HANDCRAFTED_FEATURE_NAMES = [
    "n_chars", "n_words", "avg_word_len",
    "n_urls", "n_emails", "n_phone_numbers", "n_money",
    "digit_ratio", "caps_ratio", "n_allcaps_words",
    "n_exclamation", "n_question", "punct_ratio",
    "urgency_words", "reward_words", "call_to_action_words", "credential_words",
]


def handcrafted_vector(raw: str) -> list[float]:
    """Compute the 17 hand-crafted features for one raw message."""
    import html  # local import keeps the module namespace small

    text = html.unescape(raw)
    text = pp.PLACEHOLDER_RE.sub(" ", text)
    n_chars = len(text)
    letters = len(_LETTER_RE.findall(text))
    uppers = len(_UPPER_RE.findall(text))

    # Entity counts use the same regexes as the cleaner, applied in the same
    # order (e-mails, then URLs, then money, then phone numbers), each match
    # removed before the next pattern runs so nothing is double-counted.
    n_emails = len(pp.EMAIL_RE.findall(text))
    rest = pp.EMAIL_RE.sub(" ", text)
    n_urls = len(pp.URL_RE.findall(rest))
    prose = pp.URL_RE.sub(" ", rest)       # message text without links/addresses
    n_money = len(pp.MONEY_RE.findall(prose))
    n_phone = len(pp.PHONE_RE.findall(pp.MONEY_RE.sub(" ", prose)))

    # Word-level statistics and lexicon hits are computed on the prose only,
    # so that e.g. "secure-bank.com" counts as one URL, not as two
    # credential words.
    words = _WORD_RE.findall(prose)
    n_words = len(words)
    lower_words = [w.lower().strip("'") for w in words]

    def lex(vocab: set[str]) -> int:
        return sum(w in vocab for w in lower_words)

    return [
        n_chars,
        n_words,
        (sum(map(len, words)) / n_words) if n_words else 0.0,
        n_urls,
        n_emails,
        n_phone,
        n_money,
        len(_DIGIT_RE.findall(text)) / n_chars if n_chars else 0.0,
        uppers / letters if letters else 0.0,
        len(_ALLCAPS_RE.findall(text)),
        text.count("!"),
        text.count("?"),
        len(_PUNCT_RE.findall(text)) / n_chars if n_chars else 0.0,
        lex(URGENCY_WORDS),
        lex(REWARD_WORDS),
        lex(CALL_TO_ACTION_WORDS),
        lex(CREDENTIAL_WORDS),
    ]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: array of raw strings -> (n, 17) float matrix."""

    feature_names = HANDCRAFTED_FEATURE_NAMES

    def fit(self, X, y=None):  # stateless
        return self

    def transform(self, X):
        return np.asarray([handcrafted_vector(str(x)) for x in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names, dtype=object)


def make_tfidf(remove_stopwords: bool = False, normalise: bool = True) -> TfidfVectorizer:
    """Word uni+bi-gram TF-IDF.

    ``min_df=2`` drops hapaxes (mostly typos), ``max_df=0.95`` drops tokens
    present in almost every message, ``sublinear_tf`` uses 1+log(tf) so a word
    repeated five times does not count five times as much.
    ``normalise=False`` gives the "raw" ablation baseline: scikit-learn's
    default lower-casing + ``\\b\\w\\w+\\b`` token pattern, no entity tags.
    """
    common = dict(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    if not normalise:
        return TfidfVectorizer(lowercase=True, **common)
    return TfidfVectorizer(
        preprocessor=pp.clean_text,
        tokenizer=pp.tokenize_with_stopwords if remove_stopwords else pp.tokenize,
        token_pattern=None,
        lowercase=False,
        **common,
    )


def make_handcrafted() -> Pipeline:
    """Hand-crafted features -> log1p (tames heavy-tailed counts) -> [0, 1].

    Scaling to [0, 1] (not z-scores) keeps every value non-negative, which
    Multinomial Naive Bayes requires, and puts the dense columns on the same
    scale as L2-normalised TF-IDF weights in the combined representation.
    """
    return Pipeline([
        ("extract", HandcraftedFeatures()),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", MinMaxScaler(clip=True)),
    ])


def make_features(kind: str):
    """Factory for the three feature representations compared in the study."""
    if kind == "tfidf":
        return make_tfidf()
    if kind == "handcrafted":
        return make_handcrafted()
    if kind == "combined":
        return FeatureUnion([("tfidf", make_tfidf()), ("hand", make_handcrafted())])
    raise ValueError(f"unknown feature set: {kind!r}")


FEATURE_SETS = ["tfidf", "handcrafted", "combined"]
