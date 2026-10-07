"""Automated tests for the Project 3 recommendation logic.

Run with:  python tests/test_recommender.py
"""

import csv
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import recommender  # noqa: E402

REAL_CATALOG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "raw_skills.csv"
    )
)

# Small fixture used for hand-checked TF-IDF / cosine math.
# df: alpha=1, beta=2, gamma=2, delta=1 over N=3 documents.
FIXTURE_ROWS = [
    ("Alpha Role", "Alpha, Beta"),
    ("Beta Role", "Beta, Gamma"),
    ("Gamma Role", "Gamma, Delta"),
]


def write_catalog(directory: str, rows) -> str:
    """Write a role/skills CSV inside a temp directory and return its path."""
    path = os.path.join(directory, "catalog.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "skills"])
        writer.writerows(rows)
    return path


class CatalogTestCase(unittest.TestCase):
    """Base class that provides a temporary catalog for each test."""

    rows = FIXTURE_ROWS

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.catalog_path = write_catalog(self._tempdir.name, self.rows)
        self.items = recommender.load_items(self.catalog_path)


class TestLoadItems(unittest.TestCase):
    def test_real_catalog_loads(self):
        # 1. The shipped dataset loads correctly.
        items = recommender.load_items(REAL_CATALOG)
        self.assertEqual(len(items), 10)
        for item in items:
            self.assertTrue(item["role"])
            self.assertTrue(item["skills"])
            self.assertEqual(len(item["skills"]), len(set(item["skills"])))
            for tag in item["skills"]:
                self.assertEqual(tag, tag.strip().lower())

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            recommender.load_items(os.path.join("no", "such", "catalog.csv"))

    def test_invalid_header_raises(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "bad.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("name,description\nfoo,bar\n")
            with self.assertRaises(ValueError):
                recommender.load_items(path)

    def test_empty_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "sparse.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("role,skills\nRole Without Skills,\n,Python\nValid Role,Python\n")
            items = recommender.load_items(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role"], "Valid Role")


class TestNormalizePreferences(unittest.TestCase):
    # 2. Preferences are normalized.
    def test_lowercase_and_whitespace(self):
        self.assertEqual(
            recommender.normalize_preferences("  Python,  SQL ,  ML "),
            ["python", "sql", "ml"],
        )

    def test_empty_and_blank_input(self):
        self.assertEqual(recommender.normalize_preferences(""), [])
        self.assertEqual(recommender.normalize_preferences(None), [])
        self.assertEqual(recommender.normalize_preferences("   ,  , "), [])
        self.assertEqual(recommender.normalize_preferences(", , ,"), [])

    # 7. Duplicate preferences are removed.
    def test_duplicates_are_removed(self):
        self.assertEqual(
            recommender.normalize_preferences("Python, python, SQL, python"),
            ["python", "sql"],
        )


class TestVocabularyAndUnknownSkills(CatalogTestCase):
    def test_vocabulary_is_sorted_and_unique(self):
        vocabulary = recommender.catalog_vocabulary(self.items)
        self.assertEqual(vocabulary, sorted(vocabulary))
        self.assertEqual(len(vocabulary), len(set(vocabulary)))
        self.assertIn("alpha", vocabulary)

    # 8. Unknown preferences are identified.
    def test_unknown_skills_are_reported(self):
        vocabulary = recommender.catalog_vocabulary(self.items)
        unknown = recommender.find_unknown_skills("alpha, quantum knitting", vocabulary)
        self.assertEqual(unknown, ["quantum knitting"])

    def test_known_skills_are_not_reported(self):
        vocabulary = recommender.catalog_vocabulary(self.items)
        self.assertEqual(
            recommender.find_unknown_skills("Alpha, beta", vocabulary), []
        )


class TestScoringMath(CatalogTestCase):
    def test_idf_penalises_common_tags(self):
        vocabulary, idf, _ = recommender.fit_tfidf(self.items)
        weights = dict(zip(vocabulary, idf))
        # alpha appears in 1 of 3 documents, beta in 2 of 3.
        self.assertGreater(weights["alpha"], weights["beta"])

    def test_cosine_similarity_basics(self):
        self.assertAlmostEqual(
            recommender.cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0
        )
        self.assertAlmostEqual(
            recommender.cosine_similarity([1.0, 0.0], [0.0, 5.0]), 0.0
        )
        self.assertEqual(recommender.cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        with self.assertRaises(ValueError):
            recommender.cosine_similarity([1.0], [1.0, 2.0])

    # 4. Similarity scores follow TF-IDF * cosine exactly.
    def test_score_matches_hand_computed_formula(self):
        vocabulary, idf, item_vectors = recommender.fit_tfidf(self.items)
        weights = dict(zip(vocabulary, idf))

        # User profile: only "alpha" -> single non-zero dimension.
        user_vector = recommender.vectorize(["alpha"], vocabulary, idf)
        self.assertEqual(len(user_vector), len(vocabulary))
        self.assertAlmostEqual(user_vector[vocabulary.index("alpha")], weights["alpha"])

        # Alpha Role = {alpha, beta} with TF 1/2 each.
        expected = weights["alpha"] / math.hypot(weights["alpha"], weights["beta"])
        score = recommender.cosine_similarity(user_vector, item_vectors[0])
        self.assertAlmostEqual(score, expected, places=12)

        # The raw formula: TF-IDF dot product over the product of the norms.
        item_vector = item_vectors[0]
        manual = (user_vector[0] * item_vector[0] + user_vector[1] * item_vector[1]) / (
            math.sqrt(sum(x * x for x in user_vector))
            * math.sqrt(sum(x * x for x in item_vector))
        )
        self.assertAlmostEqual(score, manual, places=12)

    def test_idf_values_follow_pdf_formula(self):
        vocabulary, idf, _ = recommender.fit_tfidf(self.items)
        weights = dict(zip(vocabulary, idf))
        self.assertAlmostEqual(weights["alpha"], math.log10(3 / 1))
        self.assertAlmostEqual(weights["beta"], math.log10(3 / 2))


class TestRecommend(CatalogTestCase):
    # 3. Matching preferences produce relevant recommendations.
    def test_matching_preferences_find_the_right_role(self):
        results = recommender.recommend(self.items, "alpha")
        self.assertTrue(results)
        self.assertEqual(results[0]["role"], "Alpha Role")
        self.assertEqual(results[0]["matched_skills"], ["alpha"])

    def test_real_catalog_relevant_result(self):
        items = recommender.load_items(REAL_CATALOG)
        results = recommender.recommend(items, "swift, kotlin, mobile")
        self.assertEqual(results[0]["role"], "Mobile App Developer")

    # 5. Results are sorted from highest to lowest score.
    def test_results_are_sorted_descending(self):
        results = recommender.recommend(self.items, "beta, gamma, delta")
        scores = [result["score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(results[0]["score"], results[-1]["score"])

    def test_scores_are_bounded(self):
        results = recommender.recommend(self.items, "alpha, beta")
        for result in results:
            self.assertGreater(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)

    # 6. The Top-N limit works.
    def test_top_n_limits_result_count(self):
        self.assertEqual(len(recommender.recommend(self.items, "beta, gamma", top_n=1)), 1)
        self.assertEqual(len(recommender.recommend(self.items, "beta, gamma", top_n=2)), 2)

    # 6b. Fewer matches than requested simply returns fewer results.
    def test_fewer_matches_than_requested(self):
        results = recommender.recommend(self.items, "alpha", top_n=10)
        self.assertEqual(len(results), 1)

    def test_invalid_top_n_returns_nothing(self):
        self.assertEqual(recommender.recommend(self.items, "alpha", top_n=0), [])
        self.assertEqual(recommender.recommend(self.items, "alpha", top_n=-3), [])

    # 7. Empty preferences are handled without crashing.
    def test_empty_preferences_return_no_results(self):
        self.assertEqual(recommender.recommend(self.items, ""), [])
        self.assertEqual(recommender.recommend(self.items, "   ,  "), [])
        self.assertEqual(recommender.recommend(self.items, None), [])

    # 8. Unknown preferences return no results (and no crash).
    def test_unknown_preferences_return_no_results(self):
        self.assertEqual(recommender.recommend(self.items, "time travel, teleportation"), [])

    def test_empty_catalog_returns_no_results(self):
        self.assertEqual(recommender.recommend([], "alpha"), [])

    # 9. Duplicate preferences do not distort the result.
    def test_duplicate_preferences_do_not_change_scores(self):
        clean = recommender.recommend(self.items, "alpha, beta")
        duplicated = recommender.recommend(self.items, "alpha, beta, alpha, beta")
        self.assertEqual(
            [(r["role"], r["score"]) for r in clean],
            [(r["role"], r["score"]) for r in duplicated],
        )

    # 10. Ordering is deterministic and independent of preference order.
    def test_ordering_is_deterministic(self):
        first = recommender.recommend(self.items, "beta, gamma, delta")
        second = recommender.recommend(self.items, "delta, gamma, beta")
        self.assertEqual(first, second)

        again = recommender.recommend(self.items, "beta, gamma, delta")
        self.assertEqual(first, again)

    def test_ties_break_alphabetically(self):
        rows = [
            ("Zeta Role", "Solo, Extra"),
            ("Alpha Role", "Solo, Extra"),
            ("Other Role", "Other, Stuff"),
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            items = recommender.load_items(write_catalog(tempdir, rows))
            results = recommender.recommend(items, "solo")
        self.assertEqual([r["role"] for r in results], ["Alpha Role", "Zeta Role"])
        self.assertEqual(results[0]["score"], results[1]["score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
