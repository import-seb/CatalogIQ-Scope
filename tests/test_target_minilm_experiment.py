"""Aggregate metrics, train-only lookup and generic packaging; no training."""
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch

from catalogiq.segment_transformer import _state_sha256
from catalogiq.target_minilm_experiment import (
    FEATURE_CONTRACT, classification_metrics, evaluate_brand_lookup,
    evaluate_minilm, fit_brand_lookup, package_model, run_experiment,
    verify_predecessors, write_aggregate_metrics,
)
from catalogiq.target_minilm_preparation import INPUT_FIELDS
import test_target_minilm_preparation as fixtures


def brand_fixture():
    features = ['Alpha', 'plain', 'plain', '', 'tie', 'tie', 'other', 'ALPHA', 'new', '', 'tie', 'other']
    labels = ['A', 'UnItemised brand', 'UnItemised brand', 'UnItemised brand', 'B', 'A', 'B',
              'A', 'UnItemised brand', 'UnItemised brand', 'B', 'B']
    frame = pd.DataFrame({'record_id': [str(i) for i in range(12)], 'group_id': [f'g{i}' for i in range(12)],
                          'ProductBrand': features, 'Brand': labels})
    return SimpleNamespace(target_column='Brand', label_names=('UnItemised brand', 'A', 'B'),
        supervised_frame=frame, train_indices=np.arange(7), validation_indices=np.arange(7, 12))


class BrandLookupTests(unittest.TestCase):
    def test_lookup_fits_only_training_rows_and_ties_use_saved_label_order(self):
        prepared = brand_fixture()
        first = fit_brand_lookup(prepared)
        self.assertEqual(first['fallback_label'], 'UnItemised brand')
        self.assertEqual(first['mapping']['tie'], 'A')
        prepared.supervised_frame.loc[7:, 'Brand'] = 'B'
        prepared.supervised_frame.loc[7:, 'ProductBrand'] = 'validation-only'
        second = fit_brand_lookup(prepared)
        for key in ('mapping', 'fallback_label', 'training_record_ids_sha256', 'training_class_counts'):
            self.assertEqual(first[key], second[key])
        self.assertNotIn('validation-only', second['mapping'])
        prepared.label_names = ('UnItemised brand', 'B', 'A')
        self.assertEqual(fit_brand_lookup(prepared)['mapping']['tie'], 'B')

    def test_all_validation_rows_catchalls_and_unseen_missing_fallback_are_scored(self):
        prepared = brand_fixture()
        result = evaluate_brand_lookup(prepared, fit_brand_lookup(prepared))
        self.assertEqual(result['full_validation']['rows'], 5)
        self.assertEqual(result['coverage_records'], 3)
        self.assertEqual(result['fallback_records'], 2)
        self.assertAlmostEqual(result['full_validation']['accuracy'], .8)
        self.assertAlmostEqual(result['full_validation']['macro_f1'], 7/9)
        self.assertEqual(result['cohorts']['missing_values']['records'], 1)
        self.assertEqual(result['cohorts']['unseen_nonmissing_values']['records'], 1)
        self.assertEqual(result['cohorts']['all_fallback_records']['metrics']['accuracy'], 1)
        self.assertTrue(result['includes_catch_all_labels'])

    def test_lookup_rejects_other_targets_or_foreign_fit_membership(self):
        prepared = brand_fixture()
        model = fit_brand_lookup(prepared)
        prepared.supervised_frame.loc[0, 'record_id'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'different training'):
            evaluate_brand_lookup(prepared, model)
        prepared.target_column = 'TargetAgeGroup'
        with self.assertRaisesRegex(ValueError, 'Brand only'):
            fit_brand_lookup(prepared)

    def test_zero_support_classes_are_retained_and_supported_macro_is_separate(self):
        result = classification_metrics([0, 1], [0, 1], ['A', 'B', 'C'])
        self.assertAlmostEqual(result['macro_f1'], 2/3)
        self.assertEqual(result['macro_f1_validation_supported_classes'], 1)
        self.assertEqual(result['per_class']['C']['support'], 0)
        self.assertEqual(np.asarray(result['confusion_matrix']).shape, (3, 3))


class TargetExperimentArtifactTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TargetMiniLMPreparationTests('test_v4_targets_select_their_own_labels_without_segment_filtering')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.prepared = self.fixture.prepare('TargetAgeGroup')

    def test_explicit_authorization_precedes_training(self):
        with self.assertRaises(PermissionError):
            run_experiment(self.prepared, self.fixture.root)

    def test_predecessor_archive_paths_roundtrip_through_json_and_checksums_are_enforced(self):
        from catalogiq.features import sha256
        archive = self.fixture.root / 'targetagegroup.zip'
        archive.write_bytes(b'previously verified artifact')
        result = {'status': 'completed_and_verified', 'handoff': {'archive': str(archive), 'archive_sha256': sha256(archive)}}
        (self.fixture.root / 'targetagegroup_result.json').write_text(json.dumps(result))
        verify_predecessors(self.fixture.root, 'Sub-Segment')
        archive.write_bytes(b'changed artifact')
        with self.assertRaisesRegex(ValueError, 'preceding target'):
            verify_predecessors(self.fixture.root, 'Sub-Segment')

    def test_metrics_require_exact_ordered_validation_membership(self):
        prepared = self.prepared
        prepared.output_directory.mkdir(parents=True)
        validation = prepared.supervised_frame.iloc[prepared.validation_indices]
        encoded = validation.TargetAgeGroup.map({label: index for index, label in enumerate(prepared.label_names)}).to_numpy()
        predictions = pd.DataFrame({'record_id': validation.record_id.tolist(),
            'true_label': validation.TargetAgeGroup.tolist(), 'predicted_label': validation.TargetAgeGroup.tolist()})
        for index in range(3):
            predictions[f'probability_{index}'] = (encoded == index).astype(float)
        path = prepared.output_directory / 'predictions.csv'
        predictions.to_csv(path, index=False)
        result = evaluate_minilm(prepared)
        self.assertEqual(result['accuracy'], 1)
        write_aggregate_metrics(prepared.output_directory, result)
        self.assertTrue((prepared.output_directory / 'confusion_matrix.csv').exists())
        predictions.iloc[::-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'ordered membership'):
            evaluate_minilm(prepared)

    def test_package_roundtrip_preserves_state_target_and_metadata_without_predictions(self):
        prepared = self.prepared
        source = prepared.output_directory
        (source / 'tokenizer').mkdir(parents=True)
        (source / 'tokenizer/tokenizer.json').write_text('{}')
        state = {'classifier.weight': torch.arange(6).reshape(3, 2).float()}
        torch.save({'state_dict': state, 'model_config': prepared.config.to_dict(), 'label_names': list(prepared.label_names)}, source / 'checkpoint.pt')
        for name in ('encoder_config.json', 'model_config.json', 'run_protocol.json', 'evaluation.json', 'label_vocabulary.json', 'feature_contract.json', 'training_manifest.json'):
            (source / name).write_text('{}')
        for name in ('per_class_metrics.csv', 'confusion_matrix.csv', 'predictions.csv'):
            (source / name).write_text('internal test artifact')
        result = {'target': 'TargetAgeGroup', 'model_identity': prepared.model_identity,
            'label_names': list(prepared.label_names), 'label_to_id': {label: index for index, label in enumerate(prepared.label_names)},
            'input_fields': list(INPUT_FIELDS), 'sequence_length': 256, 'representation': {'combined_token_budget': 256},
            'split_identity': {'allocation': 'frozen V4'}, 'config': prepared.config.to_dict(), 'feature_contract': FEATURE_CONTRACT,
            'loss_configuration': {'class_weights': None}, 'final_state_sha256': _state_sha256(state)}
        delivery = package_model(prepared, result, self.fixture.root)
        checkpoint = torch.load(Path(delivery['directory']) / 'checkpoint.pt', weights_only=True)
        self.assertEqual(checkpoint['target'], 'TargetAgeGroup')
        self.assertEqual(checkpoint['label_names'], list(prepared.label_names))
        self.assertEqual(checkpoint['sequence_length'], 256)
        with zipfile.ZipFile(delivery['archive']) as zipped:
            self.assertNotIn('predictions.csv', zipped.namelist())
            self.assertIn('source_training_manifest.json', zipped.namelist())
        self.assertTrue(delivery['checksums_and_state_verified'])


if __name__ == '__main__':
    unittest.main()
