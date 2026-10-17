# Documentation index

## Current workflow and policy

- [Cleaning guide](target_cleaning.md): command, artifacts, and caveats.
- [Structural check](structural_check.md): independent full-dataset row screening and identity contract.
- [Decision log](decisions.md): accepted and withdrawn cleaning rules.
- [Data contract](data_contract.md): current ingestion assumptions and unknowns.
- [Integration contract](integration/cleaning_contract.md): shared identity, candidate mask, and handoff artifacts.
- [Contribution workflow](../CONTRIBUTING.md): analysis-to-code process.
- [Portable Segment split workflow](findings/02_splitting/03_portable_workflow_20261010.md): authoritative cleaning/splitting commands, generated files, and membership verification.

## Evidence

- [Classical Segment validation results](findings/03_modeling/02_classical_segment_results_20261010.md): first four-field/category comparison on authoritative splits, metrics, checks and evaluator handoff.
- [Classical Segment baseline](findings/03_modeling/01_classical_segment_baseline.md): word TF-IDF, category comparison and validation artifacts on the authoritative shared partitions.
- [October 9 finalized Segment splits](findings/02_splitting/02_finalization_20261009.md): unchanged grouping and test decisions, integrity findings, and remaining risks; its local snapshot commands are historical.
- [Split leakage audit](findings/02_splitting/01_leakage_audit_20261009.md): independent duplicate checks, reviewed misses and groups, and frozen-test integrity.
- [Current combined cleaning validation](findings/01_cleaning/05_current_policy_validation.md): agreed Exclude handling, integrated structural decisions, exports and reproducibility.

- [Initial profile](findings/00_data_profile/01_profile_results.md)
- [Proposed cleaning direction](findings/00_data_profile/02_cleaning_direction.md)
- [Structural validation](findings/00_data_profile/07_structural_validation.md): counts, reproducibility and remaining assumptions.
- [Structural/main compatibility](findings/00_data_profile/08_structural_main_compatibility.md): canonical identities and checks against the merged feature pipeline.
- [Pre-Aryx findings](findings/01_cleaning/01_CatalogIQ_Pre_Aryx_Findings.md)
- [Combined cleaning validation](findings/01_cleaning/02_integration_validation.md)
- [Timestamp audit](findings/01_cleaning/03_mdm_datetime_audit.md)
- [Notebook index](../notebooks/README.md)

Findings record their analysis context; they are not executable policy. Some older
findings and notebooks predate the withdrawn J&J reassignment. The decision log and
current package define what cleaning actually does.

## Background references

- [Aryx](references/giggso/aryx_reference.md)
- [Raven](references/giggso/raven_reference.md)
- [Claude tooling](references/giggso/claude/claude_tooling.md)
- [MCP team guide](references/giggso/claude/claude_tooling_mcp_team_guide.md)

These are external context, not evidence that a dataset label should be changed.
