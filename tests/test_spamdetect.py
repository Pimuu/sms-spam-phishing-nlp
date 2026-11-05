"""Unit tests (standard-library unittest; run with `python -m unittest discover tests`)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spamdetect import config  # noqa: E402
from spamdetect.data import deduplicate, load_raw, split  # noqa: E402
from spamdetect.evaluate import bootstrap_ci, point_metrics  # noqa: E402
from spamdetect.features import (  # noqa: E402
    HANDCRAFTED_FEATURE_NAMES, HandcraftedFeatures, handcrafted_vector, make_features,
)
from spamdetect.models import make_pipeline  # noqa: E402
from spamdetect.preprocess import clean_text, tokenize  # noqa: E402


class TestPreprocess(unittest.TestCase):
    def test_entity_tokens(self):
        self.assertEqual(
            clean_text("WINNER!! Call 09061701461 or visit www.win.co.uk for £1000"),
            "winner call phonetoken or visit urltoken for moneytoken",
        )

    def test_email_before_url(self):
        self.assertIn("emailtoken", clean_text("write to john.doe@gmail.com"))
        self.assertNotIn("urltoken", clean_text("write to john.doe@gmail.com"))

    def test_short_code_and_pence(self):
        out = clean_text("Text WIN to 87121, 150p/msg")
        self.assertEqual(out, "text win to phonetoken moneytoken msg")

    def test_html_placeholder_removed(self):
        self.assertEqual(clean_text("call me on &lt;#&gt; ok"), "call me on ok")

    def test_tokenize_stopwords(self):
        toks = tokenize("please call me now u")
        self.assertEqual(toks, ["please", "call", "me", "now"])          # 1-char "u" dropped
        self.assertEqual(tokenize("please call me now", remove_stopwords=True), [])


class TestFeatures(unittest.TestCase):
    def test_handcrafted_values(self):
        f = dict(zip(HANDCRAFTED_FEATURE_NAMES,
                     handcrafted_vector("URGENT! Verify your account at http://x-bank.com now")))
        self.assertEqual(f["n_urls"], 1)
        self.assertEqual(f["n_exclamation"], 1)
        self.assertEqual(f["urgency_words"], 2)        # urgent, now
        self.assertEqual(f["credential_words"], 2)     # verify, account
        self.assertEqual(f["n_allcaps_words"], 1)

    def test_empty_message(self):
        self.assertEqual(len(handcrafted_vector("")), len(HANDCRAFTED_FEATURE_NAMES))

    def test_feature_shapes(self):
        X = np.array(["free prize call 09061701461 now", "see you at lunch", "ok"], dtype=object)
        self.assertEqual(HandcraftedFeatures().fit_transform(X).shape, (3, 17))
        Xh = make_features("handcrafted").fit_transform(X)
        self.assertTrue((Xh >= 0).all() and (Xh <= 1).all())


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.df, cls.stats = deduplicate(cls.raw)

    def test_counts(self):
        self.assertEqual(len(self.raw), config.EXPECTED_N_MESSAGES)
        self.assertEqual(self.stats["n_dedup"] + self.stats["n_duplicates_removed"], len(self.raw))
        self.assertFalse(self.df["text"].str.lower().str.split().str.join(" ").duplicated().any())

    def test_split_disjoint_and_stratified(self):
        X_tr, X_te, y_tr, y_te = split(self.df)
        self.assertEqual(len(set(X_tr) & set(X_te)), 0)
        self.assertAlmostEqual(y_tr.mean(), y_te.mean(), delta=0.01)


class TestModelsAndMetrics(unittest.TestCase):
    def test_pipeline_end_to_end(self):
        texts = np.array(["WIN a FREE prize! call 09061701461 now", "free cash txt WIN to 87121",
                          "are we still on for lunch?", "running late, see you soon"] * 10, dtype=object)
        y = np.array([1, 1, 0, 0] * 10)
        for model in ["naive_bayes", "logistic_regression", "random_forest"]:
            for feats in ["tfidf", "handcrafted", "combined"]:
                pipe = make_pipeline(model, feats).fit(texts, y)
                self.assertEqual((pipe.predict(texts) == y).mean(), 1.0, (model, feats))

    def test_metrics(self):
        y = np.array([0, 0, 1, 1]); pred = np.array([0, 1, 1, 1]); score = np.array([.1, .6, .7, .9])
        m = point_metrics(y, pred, score)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 0, 2))
        self.assertAlmostEqual(m["fpr"], 0.5)
        self.assertAlmostEqual(m["roc_auc"], 1.0)
        ci = bootstrap_ci(y, pred, score, n_boot=50)
        self.assertLessEqual(ci["f1_ci_low"], ci["f1_ci_high"])


if __name__ == "__main__":
    unittest.main()
