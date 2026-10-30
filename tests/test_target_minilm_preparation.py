"""Target-aware MiniLM inputs, with no neural initialization or training."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from catalogiq.features import sha256
from catalogiq.segment_category_ablation import build_category_texts, tokenize_category_corpus
from catalogiq.segment_transformer import ModelConfig, train_transformer
from catalogiq.target_development_preparation import MISSING_TARGETS, _support
from catalogiq.target_minilm_preparation import INPUT_FIELDS, prepare_target_minilm, tokenize_target_minilm
from test_segment_transformer import TinyTokenizer
from test_target_development_preparation import fixture


MODULE = "catalogiq.target_minilm_preparation"


class TargetMiniLMPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "frozen"
        self.source.mkdir()
        fixture(self.source)
        for role in ("train", "validation"):
            path = self.source / f"{role}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["Mnfr"] = frame.record_id.map({"a": "All others", "b": "J&J", "c": "All others",
                "d": "J&J", "e": "All others", "f": "null", "g": "All others", "h": "", "i": "none", "j": "nan"})
            frame["TargetAgeGroup"] = frame.record_id.map({"a": "Adult", "b": "Children", "c": "Adult",
                "d": "Infant", "e": "Children", "f": "", "g": "Infant", "h": "null", "i": "none", "j": "nan"})
            frame["ProductName"] = "Example product " + frame.record_id
            frame["ProductBrand"] = "Retail brand"
            frame["ProductDescription"] = "<b>Useful</b> &amp; safe <script>REMOVE</script>"
            frame["ProductContents"] = "Ingredients"
            frame["ProductCategory"] = "<b>Health</b> &gt; Care"
            frame["Category"] = "FORBIDDEN_OTHER_LABEL"
            frame.to_csv(path, index=False)
        (self.source / "model_config.json").write_text(json.dumps(ModelConfig().to_dict()), encoding="utf8")
        (self.source / "tokenizer").mkdir()
        (self.source / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf8")
        self.reseal_source()
        self.verifier = self.enterContext(patch(MODULE + ".verify_protocol_splits", return_value={"verified": True}))
        self.enterContext(patch("catalogiq.segment_transformer._initialize_torch", side_effect=AssertionError("Neural initialization forbidden")))

    def reseal_source(self):
        names = ("train.csv", "validation.csv", "protected_final_test.csv", "assignments.csv",
                 "grouping_freeze.json", "model_config.json")
        (self.source / "completion.json").write_text(json.dumps({"format": "catalogiq-portable-split-v1",
            "artifact_sha256": {name: sha256(self.source / name) for name in names}}), encoding="utf8")

    def proposal(self, target, *, reverse=False):
        directory = self.root / target
        directory.mkdir()
        roles = [pd.read_csv(self.source / f"{role}.csv", dtype=str, keep_default_na=False) for role in ("train", "validation")]
        frame = pd.concat(roles, ignore_index=True).sort_values("record_id").reset_index(drop=True)
        frame["original_split"] = frame.split
        frame["is_labeled"] = ~frame[target].str.strip().str.casefold().isin(MISSING_TARGETS)
        support, _, _ = _support(frame, target)
        frame.loc[frame.group_id.eq("g3"), "split"] = "train"
        if reverse:
            frame = frame.iloc[::-1]
        frame[["record_id", "group_id", "original_split", target, "is_labeled", "split"]].to_csv(directory / "development_assignments.csv", index=False)
        protected = pd.read_csv(self.source / "protected_final_test.csv", dtype=str, keep_default_na=False)
        pd.concat([frame[["record_id", "group_id", "split"]], protected[["record_id", "group_id", "split"]]], ignore_index=True).to_csv(directory / "assignments.csv", index=False)
        protected.to_csv(directory / "protected_final_test.csv", index=False)
        # Deliberately non-alphabetic: the adapter must preserve this class order.
        support.iloc[::-1].to_csv(directory / "class_coverage.csv", index=False)
        source_seal = json.loads((self.source / "completion.json").read_text(encoding="utf8"))
        (directory / "protocol.json").write_text(json.dumps({"version": f"{target.casefold()}-development-allocation-v1",
            "status": "proposed; pending review", "preparation": {"target_column": target,
            "completion_sha256": sha256(self.source / "completion.json"),
            "consumed_artifact_sha256": source_seal["artifact_sha256"]}}), encoding="utf8")
        self.reseal_proposal(directory)
        return directory

    def reseal_proposal(self, directory):
        names = ("protocol.json", "assignments.csv", "development_assignments.csv", "class_coverage.csv", "protected_final_test.csv")
        (directory / "completion.json").write_text(json.dumps({"artifact_sha256": {
            name: sha256(directory / name) for name in names}}), encoding="utf8")

    def prepare(self, target, directory=None, **kwargs):
        if directory is not None:
            kwargs.update(allocation_directory=directory, allocation_completion_sha256=sha256(directory / "completion.json"))
        return prepare_target_minilm(self.source, target, self.root / "models", **kwargs)

    def test_v4_targets_select_their_own_labels_without_segment_filtering(self):
        for target, names in (("Mnfr", ("All others", "J&J")), ("TargetAgeGroup", ("Adult", "Children", "Infant"))):
            with self.subTest(target=target):
                result = self.prepare(target)
                self.assertEqual(result.label_names, names)
                self.assertEqual(len(result.frame), 10)
                self.assertEqual(len(result.supervised_frame), 6)
                self.assertTrue(result.frame.split.equals(result.frame.original_split))
                self.assertEqual(set(result.supervised_frame.iloc[result.train_indices].record_id), {"a", "b", "g"})
                self.assertEqual(set(result.supervised_frame.iloc[result.validation_indices].record_id), {"c", "d", "e"})
                self.assertNotIn("Segment", result.frame)
        self.verifier.assert_called_with(self.source.resolve(), verify_exports=False)

    def test_approved_allocations_preserve_record_group_and_label_order(self):
        for target in ("Brand", "Sub-Segment"):
            with self.subTest(target=target):
                directory = self.proposal(target, reverse=True)
                result = self.prepare(target, directory)
                self.assertEqual(result.frame.record_id.tolist(), list("jihgfedcba"))
                self.assertEqual(result.label_names, ("Rare", "Common"))
                self.assertEqual(result.supervised_frame.record_id.tolist(), list("edcba"))
                self.assertEqual(result.legacy_frame.Segment.tolist(), ["Rare", "Rare", "Common", "Common", "Common"])
                self.assertEqual(set(result.frame.loc[result.frame.group_id.eq("g3"), "split"]), {"train"})
                corpus = tokenize_target_minilm(result, tokenizer=TinyTokenizer())
                self.assertEqual(corpus.label_ids.tolist(), [0, 0, 1, 1, 1])
                self.assertEqual(corpus.record_ids, tuple("edcba"))

    def test_representation_and_trainer_signature_match_legacy_without_training(self):
        prepared = self.prepare("Mnfr")
        before = prepared.frame.copy(deep=True)
        corpus = tokenize_target_minilm(prepared, tokenizer=TinyTokenizer())
        expected = tokenize_category_corpus(prepared.legacy_frame, prepared.label_names, prepared.config, tokenizer=TinyTokenizer())
        self.assertEqual(corpus.text_sha256, expected.text_sha256)
        self.assertEqual(corpus.token_sha256, expected.token_sha256)
        self.assertEqual(corpus.sequence_length, 256)
        text = build_category_texts(prepared.legacy_frame, prepared.config)[0]
        self.assertTrue(text.startswith("[ProductCategory] Health > Care [ProductName]"))
        self.assertNotIn("REMOVE", text)
        self.assertNotIn("FORBIDDEN", text)
        self.assertEqual(tuple(prepared.legacy_frame.columns), ("record_id", "Segment", *INPUT_FIELDS))
        arguments = prepared.trainer_arguments(corpus)
        inspect.signature(train_transformer).bind(**arguments)
        self.assertEqual(arguments["output_dir"], prepared.output_directory)
        pd.testing.assert_frame_equal(prepared.frame, before)

    def test_local_tokenizer_loading_never_requests_model_weights(self):
        prepared = self.prepare("Mnfr")
        tokenizer_factory = Mock(return_value=TinyTokenizer())
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_factory),
                               AutoModel=Mock(side_effect=AssertionError("Weights forbidden")))
        with patch.dict("sys.modules", {"transformers": fake}):
            tokenize_target_minilm(prepared)
        tokenizer_factory.assert_called_once_with(str(prepared.tokenizer_directory), local_files_only=True)
        fake.AutoModel.assert_not_called()

    def test_target_identities_vocabularies_and_directories_are_distinct_without_writes(self):
        directories = {target: self.proposal(target) for target in ("Brand", "Sub-Segment")}
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        results = [self.prepare(target, directories.get(target)) for target in ("Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment")]
        self.assertEqual(len({result.model_identity for result in results}), 4)
        self.assertEqual(len({result.output_directory for result in results}), 4)
        self.assertFalse((self.root / "models").exists())
        again = self.prepare("Mnfr")
        self.assertEqual(again.model_identity, results[0].model_identity)
        self.assertEqual(before, {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_label_order_can_be_explicit_for_v4_but_cannot_change_approved_class_order(self):
        prepared = self.prepare("Mnfr", label_names=("J&J", "All others"))
        self.assertEqual(prepared.label_names, ("J&J", "All others"))
        corpus = tokenize_target_minilm(prepared, tokenizer=TinyTokenizer())
        self.assertEqual(corpus.label_ids[:2].tolist(), [1, 0])
        directory = self.proposal("Brand")
        with self.assertRaisesRegex(ValueError, "class ordering"):
            self.prepare("Brand", directory, label_names=("Common", "Rare"))

    def test_unsupported_targets_wrong_allocation_and_missing_approval_fail(self):
        for target in ("Platform", "Segment", "Unknown"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.prepare(target)
        with self.assertRaisesRegex(ValueError, "approved allocation"):
            self.prepare("Brand")
        directory = self.proposal("Brand")
        with self.assertRaisesRegex(ValueError, "retain frozen V4"):
            self.prepare("Mnfr", directory)
        with self.assertRaisesRegex(ValueError, "target or frozen source"):
            self.prepare("Sub-Segment", directory)
        with self.assertRaisesRegex(ValueError, "approved allocation"):
            prepare_target_minilm(self.source, "Brand", self.root / "models", allocation_directory=directory,
                                   allocation_completion_sha256="0" * 64)

    def test_group_isolation_and_protected_records_including_missing_targets(self):
        for column, value in (("group_id", "gz"), ("record_id", "z"), ("group_id", "g1")):
            with self.subTest(column=column, value=value):
                path = self.source / "validation.csv"
                original = path.read_bytes()
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame.loc[frame.record_id.eq("f"), column] = value
                frame.to_csv(path, index=False)
                self.reseal_source()
                with self.assertRaisesRegex(ValueError, "isolation"):
                    self.prepare("Mnfr")
                path.write_bytes(original)
                self.reseal_source()

    def test_allocation_labels_or_group_ids_cannot_be_substituted_even_if_resealed(self):
        directory = self.proposal("Brand")
        path = directory / "development_assignments.csv"
        original = path.read_bytes()
        for column, value in (("Brand", "Changed"), ("group_id", "Changed"), ("is_labeled", "False")):
            with self.subTest(column=column):
                path.write_bytes(original)
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                frame.loc[0, column] = value
                frame.to_csv(path, index=False)
                self.reseal_proposal(directory)
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.prepare("Brand", directory)

    def test_test_csv_is_never_opened_or_hashed_and_protected_columns_are_identity_only(self):
        directory = self.proposal("Brand")
        original_read, original_hash = pd.read_csv, sha256
        def guarded_read(path, *args, **kwargs):
            self.assertNotEqual(Path(path).name, "test.csv")
            if Path(path).name == "protected_final_test.csv":
                self.assertEqual(kwargs["usecols"], ["record_id", "group_id", "split"])
            self.assertNotIn("Category", kwargs["usecols"])
            return original_read(path, *args, **kwargs)
        def guarded_hash(path):
            self.assertNotEqual(Path(path).name, "test.csv")
            return original_hash(path)
        with patch(MODULE + ".pd.read_csv", side_effect=guarded_read), patch(MODULE + ".sha256", side_effect=guarded_hash):
            self.prepare("Brand", directory)
            self.prepare("Mnfr")

    def test_configuration_drift_and_reused_output_paths_are_rejected(self):
        for config in (ModelConfig(), ModelConfig(sequence_length=512), ModelConfig(sequence_length=256, local_files_only=False),
                       ModelConfig(sequence_length=256, field_character_limits=(1, 2, 3, 4))):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "256 tokens"):
                self.prepare("Mnfr", config=config)
        prepared = self.prepare("Mnfr")
        prepared.output_directory.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.prepare("Mnfr")
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_target_minilm(self.source, "Mnfr", self.source / "models")

    def test_foreign_corpus_or_mutated_group_roles_are_not_trainer_compatible(self):
        prepared = self.prepare("Mnfr")
        corpus = tokenize_target_minilm(prepared, tokenizer=TinyTokenizer())
        with self.assertRaisesRegex(ValueError, "Corpus does not match"):
            prepared.trainer_arguments(replace(corpus, record_ids=tuple(reversed(corpus.record_ids))))
        changed = prepared.frame.copy()
        changed.loc[changed.record_id.eq("a"), "group_id"] = "different"
        with self.assertRaisesRegex(ValueError, "Corpus does not match"):
            replace(prepared, frame=changed).trainer_arguments(corpus)


if __name__ == "__main__":
    unittest.main()
