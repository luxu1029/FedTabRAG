#!/usr/bin/env python3
"""Synthetic checks for exact multi-positive retrieval metrics."""

import math
import unittest

from src.retrieval.evaluate_retrieval import metrics_from_ranks


class RetrievalMetricsTest(unittest.TestCase):
    def test_single_positive(self):
        metrics = metrics_from_ranks([5], [1, 5, 10])
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@5"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.2)
        self.assertAlmostEqual(metrics["ap"], 0.2)

    def test_multiple_positives(self):
        metrics = metrics_from_ranks([1, 3], [1, 5, 10])
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["ap"], (1.0 + 2.0 / 3.0) / 2.0)
        expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(metrics["ndcg@10"], expected_ndcg)

    def test_ap_is_bounded_for_unique_tie_broken_ranks(self):
        metrics = metrics_from_ranks([1, 2, 3], [1, 5, 10])
        self.assertLessEqual(metrics["ap"], 1.0)
        self.assertAlmostEqual(metrics["ap"], 1.0)


if __name__ == "__main__":
    unittest.main()
