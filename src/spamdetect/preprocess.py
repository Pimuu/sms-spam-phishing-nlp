"""Text preprocessing: cleaning, entity normalisation and tokenisation.

Design notes
------------
* **Entity normalisation.** Spam is full of strings like
  ``www.win-82050.co.uk``, ``09061701461`` or ``£1500``. These are near-unique
  as surface strings and would each get their own sparse TF-IDF column. What
  predicts spam is the *type* of entity, so URLs, e-mail addresses, phone
  numbers, money amounts and other numbers are replaced by placeholder tokens
  (``urltoken``, ``phonetoken``, ...) before tokenisation.
* **No default stop-word removal.** scikit-learn's English stop-word list
  contains *call*, *now*, *please* and *get*: call-to-action words that are
  among the strongest spam cues in SMS. TF-IDF
  already down-weights uninformative frequent words, so we keep all tokens.
  The effect of stop-word removal is measured in the ablation study.
* **Case is discarded here** (lower-casing). The shouting signal is not lost:
  the hand-crafted features in ``features.py`` work on the *raw* text and keep
  capitalisation, punctuation and character statistics.
"""
from __future__ import annotations

import html
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------------------------------------------------
# Regular expressions (compiled once). Order of application matters: e-mails
# are replaced before URLs (an address contains a domain), and both before
# money/phone/number patterns so that digits inside them are not turned into
# separate number tokens.
# --------------------------------------------------------------------------
URL_RE = re.compile(
    r"""(
        (?:https?://|www\.)\S+                                   # explicit scheme / www
      | \b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*                     # bare domain ...
        \.(?:com|net|org|biz|info|co\.uk|uk|tv|ac|me|mobi|ly|io|xyz|top|site|online|link|app|cc|us)\b  # common TLD
        (?:/\S*)?
    )""",
    re.IGNORECASE | re.VERBOSE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Currency: a symbol followed by an amount, or an amount followed by p/pence.
MONEY_RE = re.compile(
    r"(?:[£$€]\s?\d[\d,]*(?:\.\d+)?)|(?:\b\d+(?:\.\d+)?\s?(?:p|pence|gbp|usd|pounds?)\b)",
    re.IGNORECASE,
)
# Phone numbers: 7+ consecutive digits, optionally separated by single spaces
# or dashes (UK premium numbers like 09061701461), or 5-digit SMS short codes
# (e.g. "text WIN to 87121") which are ubiquitous in UK SMS spam.
PHONE_RE = re.compile(r"(?:\+?\d[\d \-]{5,}\d)|(?:\b\d{5}\b)")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z]+")

# The dataset encodes a redacted placeholder as "&lt;#&gt;" (i.e. "<#>")
# and "&lt;DECIMAL&gt;" etc.; unescaping turns these into "<#>" / "<DECIMAL>".
PLACEHOLDER_RE = re.compile(r"<\s*[#a-z]*\s*>", re.IGNORECASE)

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Normalise one raw message into a lower-cased, entity-tagged string.

    >>> clean_text("WINNER!! Call 09061701461 or visit www.win.co.uk for £1000")
    'winner call phonetoken or visit urltoken for moneytoken'
    """
    text = html.unescape(text)                     # &amp; -> &, &lt; -> <
    text = PLACEHOLDER_RE.sub(" ", text)           # dataset's anonymisation tags
    text = EMAIL_RE.sub(" emailtoken ", text)      # before URLs: "a@b.com" contains a domain
    text = URL_RE.sub(" urltoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)             # drop punctuation/symbols
    return WS_RE.sub(" ", text).strip()


def tokenize(text: str, remove_stopwords: bool = False, min_len: int = 2) -> list[str]:
    """Split a *cleaned* string into alphabetic tokens.

    Single characters ("u", "r", "n") are dropped by default: they are SMS
    shorthand present in both classes and add noise.
    """
    tokens = [t for t in TOKEN_RE.findall(text) if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def tokenize_with_stopwords(text: str) -> list[str]:
    """Picklable tokenizer variant used by the stop-word ablation."""
    return tokenize(text, remove_stopwords=True)
