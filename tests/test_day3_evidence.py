#!/usr/bin/env python3
"""Regression checks for generated Day-3 evidence artifacts."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Day3EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(
            (ROOT / "outputs/audit/day3_evidence_audit.json").read_text(encoding="utf-8")
        )
        cls.mapping = json.loads(
            (ROOT / "outputs/audit/evidence_mapping_report.json").read_text(encoding="utf-8")
        )
        cls.schema = json.loads(
            (ROOT / "configs/data/evidence_schema.json").read_text(encoding="utf-8")
        )

    def test_cell_text_multisets_are_identical(self):
        self.assertEqual(self.audit["cell_multiset_mismatch_count"], 0)

    def test_no_label_leakage_or_duplicate_ids(self):
        self.assertTrue(all(value == 0 for value in self.audit["leakage_counts"].values()))
        self.assertTrue(
            all(value == 0 for value in self.audit["duplicate_evidence_ids"].values())
        )

    def test_all_positive_references_exist(self):
        self.assertEqual(self.audit["bad_reference_count"], 0)

    def test_mapping_and_manual_sample_are_complete(self):
        self.assertEqual(self.mapping["query_count"], 8281)
        self.assertEqual(self.mapping["unmapped_variant_records"], 0)
        self.assertEqual(self.audit["inspection_count"], 50)

    def test_schema_declares_structure_boundaries(self):
        constraints = self.schema["constraints"]
        self.assertTrue(constraints["flat_has_no_gt_row_boundaries"])
        self.assertTrue(constraints["predicted_text_placement_uses_row_major_only"])


if __name__ == "__main__":
    unittest.main()
