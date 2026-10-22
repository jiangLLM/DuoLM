import math
from types import SimpleNamespace
import unittest

from duolm.next_item_metrics import ranking_metrics


def query(user, candidates=("b", "a"), positive=0, query_id=None):
    return SimpleNamespace(query_id=query_id or user, user_id=user,
                           candidate_ids=candidates, positive_index=positive)


class RankingMetricTests(unittest.TestCase):
    def test_retrieval_misses_and_explicit_unavailable_users_remain_in_denominator(self):
        result = ranking_metrics([(query("a"), [2, 1]), (query("b", positive=None), [2, 1])],
                                 expected_user_ids=["a", "b", "c"], unavailable_user_ids=["c"])
        self.assertAlmostEqual(result["ndcg@10"], 1 / 3)
        self.assertAlmostEqual(result["retrieval_recall"], 1 / 3)
        self.assertEqual((result["users"], result["queries"], result["unavailable_users"]), (3, 2, 1))

    def test_tie_break_is_item_identity_not_candidate_order(self):
        first = ranking_metrics([(query("u"), [0, 0])], expected_user_ids=["u"])
        second = ranking_metrics([(query("u", candidates=("a", "b"), positive=1), [0, 0])], expected_user_ids=["u"])
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["ndcg@10"], 1 / math.log2(3))

    def test_macro_by_user_not_query(self):
        result = ranking_metrics([(query("u", query_id="u1"), [2, 1]),
                                  (query("u", positive=None, query_id="u2"), [2, 1]),
                                  (query("v"), [2, 1])], expected_user_ids=["u", "v"])
        self.assertEqual(result["hr@10"], 0.75)

    def test_top_k_and_mrr(self):
        q = query("u", candidates=("a", "b", "c"), positive=2)
        result = ranking_metrics([(q, [3, 2, 1])], expected_user_ids=["u"], k=2)
        self.assertEqual(result["mrr@2"], 0)
        self.assertEqual(result["retrieval_recall"], 1)

    def test_missing_undeclared_users_cannot_disappear(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ranking_metrics([(query("u"), [2, 1])], expected_user_ids=["u", "v"])

    def test_all_unavailable_is_zero(self):
        result = ranking_metrics([], expected_user_ids=["u"], unavailable_user_ids=["u"])
        self.assertEqual(result["ndcg@10"], 0)

    def test_bad_scores_duplicates_and_unavailable_overlap_rejected(self):
        for rows in ([(query("u"), [float("nan"), 0])], [(query("u"), [0])],
                     [(query("u"), [0, 1]), (query("u"), [0, 1])]):
            with self.assertRaises(ValueError):
                ranking_metrics(rows, expected_user_ids=["u"])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ranking_metrics([(query("u"), [0, 1])], expected_user_ids=["u"], unavailable_user_ids=["u"])


if __name__ == "__main__":
    unittest.main()
