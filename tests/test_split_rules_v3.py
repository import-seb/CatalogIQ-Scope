"""Target-free rule refinement behavior on synthetic products and families."""
from dataclasses import replace
import unittest

import pandas as pd

from catalogiq.split_rule_candidates_v3 import RuleCandidateConfig
from catalogiq.split_rule_evidence_v3 import comparison_name, prepare_rule_products_v3, remove_negative_claims
from catalogiq.split_rules_v3 import RuleRefinementConfig, pair_evidence_v3, refine_rule_groups_v3
from catalogiq.splitting import SplitConfig


class RuleRefinementTests(unittest.TestCase):
    def setUp(self):
        self.split_config = SplitConfig(grouping_version=2)
        self.config = RuleRefinementConfig()

    def run_groups(self, names, brands=None, extra=None, ids=None, config=None):
        frame = pd.DataFrame({"ProductName": names, "ProductBrand": brands or ["Willow"] * len(names)})
        if extra:
            for key, values in extra.items():
                frame[key] = values
        ids = ids or [f"record{i:03}" for i in range(len(frame))]
        return refine_rule_groups_v3(frame, ids, self.split_config, config or self.config)

    def test_pack_variants_word_order_and_units(self):
        left = comparison_name("Willow Mint Lozenges, 38 Lozenges Each (Pack of 3)")
        right = comparison_name("3 Pack Willow Mint Lozenges 38 Count")
        self.assertNotIn("38", left)
        self.assertNotIn("38", right)
        groups, _, _ = self.run_groups(["Willow Mint Strong Lozenges, 38 Lozenges Each (Pack of 3)",
                                        "3 Pack Willow Strong Mint Lozenges 38 Count"])
        self.assertEqual(groups[0], groups[1])
        self.assertIn("500 mg", comparison_name("Willow Named Herb 500mg, 2x500ml Bottles"))
        self.assertNotIn("500 ml", comparison_name("Willow Named Herb 500mg, 2x500ml Bottles"))

    def test_negated_coordinated_fillers_not_product_evidence(self):
        frame = pd.DataFrame({"ProductBrand": ["Willow"] * 2, "ProductName": [
            "Willow Zorvex 365 Capsules, Lab Verified No Magnesium Or Rice Fillers",
            "Willow Milk Thistle 365 Capsules, Lab Verified No Magnesium Or Rice Fillers"]})
        products, _ = prepare_rule_products_v3(frame, self.split_config, self.config)
        for product in products:
            self.assertNotIn("magnesium", product.core)
            self.assertNotIn("rice", product.core)
        self.assertIsNone(pair_evidence_v3(*products, self.config))
        groups, _, _ = refine_rule_groups_v3(frame, ["a", "b"], self.split_config)
        self.assertNotEqual(groups[0], groups[1])

    def test_negative_claim_boundaries_preserve_positive_formula(self):
        for boundary in (", ", "; ", " | ", " - ", " – ", " — "):
            name = comparison_name("Willow No Magnesium Or Rice Fillers" + boundary + "Milk Thistle Capsules")
            self.assertIn("milk thistle", name)
            self.assertNotIn("magnesium", name)
            self.assertNotIn("rice", name)
        self.assertIn("with Milk Thistle", remove_negative_claims("No rice fillers with Milk Thistle"))
        self.assertIn("magnesium citrate", comparison_name("Willow Magnesium Citrate, Sugar-Free"))
        name = comparison_name("Willow No Artificial Colors Zinc Tablets")
        self.assertIn("zinc", name)
        self.assertNotIn("colors", name)
        for claim in ("No fillers", "No preservatives", "Without artificial flavors"):
            self.assertIn("zinc", comparison_name("Willow " + claim + " Zinc Tablets"))
        name = comparison_name("Willow No Rice Fillers Or Magnesium Fillers Zinc Tablets")
        self.assertIn("zinc", name)
        self.assertNotIn("rice", name)
        self.assertNotIn("magnesium", name)

    def test_exact_branded_formulation_line_can_have_empty_core(self):
        frame = pd.DataFrame({"ProductBrand": ["Willow"] * 2, "ProductName": [
            "Willow Maximum Strength Softgels, 36 Each",
            "Willow Maximum Strength Softgels, 36 ea (Pack of 2)"]})
        products, _ = prepare_rule_products_v3(frame, self.split_config, self.config)
        self.assertFalse(products[0].core)
        groups, edges, _ = refine_rule_groups_v3(frame, ["a", "b"], self.split_config)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(edges[0][2], "exact_branded_generic_line")

    def test_generic_same_brand_words_do_not_establish_approximate_family(self):
        groups, _, _ = self.run_groups([
            "Willow Plant Based Valerian Root Sleep Support Capsules",
            "Willow Plant Based Glucosamine Joint Support Tablets"])
        self.assertNotEqual(groups[0], groups[1])
        groups, _, _ = self.run_groups(["Willow Wellness Dietary Supplement Capsules"] * 2,
                                        extra={"ProductDescription": ["first generic description", "second generic description"]})
        self.assertNotEqual(groups[0], groups[1])

    def test_distinctive_named_line_variants_keep_different_formulations_together(self):
        groups, edges, _ = self.run_groups([
            "Willow Northern Edge Vitamin C Tablets, 30 Count",
            "Willow Northern Edge Vitamin D3 Gummies, 60 Count"])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(edges[0][2], "distinctive_named_line")

    def test_same_active_line_can_change_delivery_form(self):
        groups, _, _ = self.run_groups([
            "Willow Lidocaine Maximum Strength Patch, 5 Count",
            "Willow Lidocaine Pain Relief Spray, 4 Oz"])
        self.assertEqual(groups[0], groups[1])

    def test_shared_mineral_and_dose_need_distinctive_line_for_different_salts(self):
        groups, _, _ = self.run_groups([
            "Willow Magnesium Citrate 200mg Capsules", "Willow Magnesium Oxide 200mg Capsules"])
        self.assertNotEqual(groups[0], groups[1])
        groups, _, _ = self.run_groups([
            "Willow UltraMag Magnesium Citrate 200mg Capsules", "Willow UltraMag Magnesium Oxide 200mg Capsules"])
        self.assertEqual(groups[0], groups[1])

    def test_branded_generic_lines_require_same_canonical_brand(self):
        groups, _, _ = self.run_groups([
            "Maximum Strength Softgels, 36 Each", "Maximum Strength Softgels, 72 Each"],
            brands=["Willow", "Aspen"])
        self.assertNotEqual(groups[0], groups[1])

    def test_empty_records_and_brand_only_records_do_not_become_one_family(self):
        groups, edges, _ = self.run_groups(["", ""], brands=["Willow", "Willow"])
        self.assertNotEqual(groups[0], groups[1])
        self.assertFalse(edges)

    def test_identical_short_unbranded_classifier_payload_is_kept_together(self):
        groups, edges, _ = self.run_groups(["Mint", "Mint"], brands=["", ""],
            extra={"ProductDescription": ["A named, precise item."] * 2, "ProductContents": ["Leaf extract."] * 2})
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(edges[0][2], "exact_classifier_payload")
        groups, _, _ = self.run_groups(["", ""], brands=["", ""],
                                       extra={"ProductDescription": ["Precise copied product information."] * 2})
        self.assertEqual(groups[0], groups[1])

    def test_common_identifier_without_product_evidence_is_not_authority(self):
        groups, _, _ = self.run_groups(["Willow Zorvex Tablets", "Willow Quelorin Tablets"],
                                      extra={"Upc": ["036000291452"] * 2})
        self.assertNotEqual(groups[0], groups[1])

    def test_scientific_notation_identifier_is_never_reconstructed(self):
        groups, _, stats = self.run_groups(["Willow Zorvex Tablets", "Willow Quelorin Tablets"],
                                           extra={"Upc": ["3.60003E+10"] * 2})
        self.assertEqual(stats["valid_gtin_rows"], 0)
        self.assertNotEqual(groups[0], groups[1])

    def test_long_substantive_cross_brand_template_can_prevent_leakage(self):
        template = " organic mullein garlic oil for ear infections, eardrops for swimmers ear and wax removal, adults and kids and babies"
        groups, _, _ = self.run_groups(["Willow" + template, "Aspen" + template], brands=["Willow", "Aspen"])
        self.assertEqual(groups[0], groups[1])
        groups, _, _ = self.run_groups(["Willow cooling soft cushion", "Aspen cooling soft cushion"],
                                       brands=["Willow", "Aspen"])
        self.assertNotEqual(groups[0], groups[1])

    def test_repeated_supplier_body_is_filtered_without_erasing_large_line(self):
        frame = pd.DataFrame({"ProductBrand": ["Willow"] * 12, "ProductName": [
            f"Willow Formula{i} Unique{i} Capsules Trustworthy Excellence Laboratory" for i in range(12)]})
        products, stats = prepare_rule_products_v3(frame, self.split_config, self.config)
        self.assertEqual(stats["brands_with_repeated_body_template"], 1)
        self.assertTrue(all("excellence" not in p.core for p in products))
        repeated_line = frame.copy()
        repeated_line["ProductName"] = [f"Willow Northern Edge Capsules Color{i}" for i in range(12)]
        products, stats = prepare_rule_products_v3(repeated_line, self.split_config, self.config)
        self.assertTrue(all({"northern", "edge"}.issubset(p.core) for p in products))
        # A nonempty template stats dictionary must coexist with nested candidate
        # metadata; Counter arithmetic over those dictionaries would fail.
        _, _, full_stats = refine_rule_groups_v3(frame, [str(i) for i in range(12)], self.split_config)
        self.assertIsInstance(full_stats["config"], dict)

    def test_anchor_intersection_blocks_disjoint_family_chain(self):
        groups, _, stats = self.run_groups([
            "Willow Amber Beta", "Willow Amber Beta Gamma Delta", "Willow Gamma Delta"])
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertGreater(stats.get("component_family_anchor_rejections", 0), 0)

    def test_multiple_representatives_reject_common_word_weak_chain(self):
        groups, _, stats = self.run_groups([
            "Willow Anchor Amber", "Willow Anchor Amber Gamma", "Willow Anchor Gamma"])
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        self.assertGreater(stats.get("component_representative_rejections", 0), 0)

    def test_large_legitimate_family_has_no_arbitrary_size_cap(self):
        names = [f"Willow Lumora 500 mg Tablets, {i + 1} Count" for i in range(130)]
        groups, edges, stats = self.run_groups(names)
        self.assertEqual(len(set(groups)), 1)
        self.assertEqual(stats["largest_group"], 130)
        self.assertFalse(stats["group_size_cap_applied"])
        self.assertTrue(edges)

    def test_row_order_and_targets_cannot_change_components_or_edge_decisions(self):
        frame = pd.DataFrame({"ProductName": ["Willow Northern Edge 30 Tablets", "Willow Northern Edge 60 Tablets",
                                               "Willow Southern Ridge Tablets"],
                              "ProductBrand": ["Willow"] * 3, "Segment": ["one", "two", "three"],
                              "Family": ["label one", "label two", "label three"], "split": ["train", "test", "validation"]})
        ids = ["c", "a", "b"]
        groups, edges, _ = refine_rule_groups_v3(frame, ids, self.split_config)
        changed = frame.iloc[[2, 0, 1]].copy()
        changed["Segment"] = "totally changed target"
        changed["Family"] = "changed label"
        changed["split"] = "test"
        changed_ids = [ids[i] for i in [2, 0, 1]]
        other_groups, other_edges, _ = refine_rule_groups_v3(changed, changed_ids, self.split_config)
        self.assertEqual(dict(zip(ids, groups)), dict(zip(changed_ids, other_groups)))
        named_edges = lambda es, rowids: [(rowids[a], rowids[b], reason, score) for a, b, reason, score in es]
        self.assertEqual(named_edges(edges, ids), named_edges(other_edges, changed_ids))

    def test_empty_input_invalid_ids_and_config_roundtrip(self):
        groups, edges, stats = refine_rule_groups_v3(pd.DataFrame(), [], self.split_config)
        self.assertEqual(len(groups), 0)
        self.assertEqual(edges, [])
        self.assertEqual(stats["largest_group"], 0)
        self.assertEqual(RuleRefinementConfig.from_dict(self.config.to_dict()), self.config)
        for invalid in (replace(self.config, candidate=RuleCandidateConfig(max_character_comparisons=5)),):
            self.assertEqual(RuleRefinementConfig.from_dict(invalid.to_dict()), invalid)
        with self.assertRaises(ValueError):
            self.run_groups(["one", "two"], ids=["same", "same"])
        with self.assertRaises(ValueError):
            RuleRefinementConfig(component_representatives=0)
        with self.assertRaises(ValueError):
            RuleRefinementConfig(template_document_fraction=float("nan"))


if __name__ == "__main__":
    unittest.main()
