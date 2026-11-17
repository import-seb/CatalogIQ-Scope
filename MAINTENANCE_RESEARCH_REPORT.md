**Keep the finished MiniLM models. Add conservative, versioned correction memory first; use small head-only updates only after real confirmed-data and retention checks. LoRA is not currently justified as the default maintenance method.**

Publication note (November 15, 2026): private `data/processed/...` evidence referenced below is excluded from the PR. These historical findings are unchanged; the [shareable verification summary](docs/maintenance_verification.md) and [portable test instructions](experiments/maintenance_research/README.md) cover the published engineering subset. Further model-quality research remains deferred.

Completed 14 November 2026 (America/New_York). No production behavior, original model, source dataset, split, inference adapter or original environment was changed. Nothing was pushed, merged or deployed.

Repository cleanup removed **505 abandoned replacement-research files (137.97 MiB)**: both pretrained-MiniLM SGD experiments, replacement-specific LoRA sources/checkpoints, copied samples, logs and obsolete reports. Reusable resource limits, explicit-CPU LoRA reloading and provenance were consolidated into [one maintenance directory](experiments/maintenance_research/README.md). The shared verified dependency overlay and all original/shared or uncertain-ownership artifacts were retained. Exact pre-deletion inventory (`cleanup_inventory.json`, private local inventory excluded from the PR).

After verification, another **101 redundant research intermediates (392.04 MiB)** were removed: full screening embeddings, duplicate CPU-profiling model files and temporary caches. Four compact, bitwise-verified reusable caches and final small maintenance deltas remain in one ignored evidence tree (approximately 117.7 MiB before final report tables/seal). No backup directory replaced the clutter.

**Data validity is the main limitation.** No verified incoming human-confirmation ledger was available. Every update used previously trained-on records, explicitly labeled as proxiesâ€”not genuinely unseen products or newly adjudicated corrections. Original trainer membership hashes were reproduced from development-only exports; protected test features, labels and identity registries stayed closed. Validation membership was preserved separately for each target, including Sub-Segmentâ€™s existing target-specific allocation.

| Existing model | Seen training records | Fixed validation records | Original accuracy | Original macro-F1 | Proxy errors before update |
|---|---:|---:|---:|---:|---:|
| Segment | 55,776 | 11,952 | 96.9461% | 0.944490 | 172 / 512 |
| Mnfr | 55,233 | 11,812 | 99.9661% | 0.995729 | 6 / 512 |
| TargetAgeGroup | 54,282 | 11,624 | 99.7505% | 0.987426 | 25 / 512 |
| Sub-Segment | 53,956 | 11,562 | 88.5833% | 0.657268 | 157 / 512 |

Each neural study used the **existing fully fine-tuned checkpoint and learned classification head**. Nothing was pretrained, rebuilt through training, or fitted as a scratch replacement classifier. Each proxy update contained 512 records plus 2,048 disjoint whole-group historical replay records, two epochs, and three fixed orders: shuffle, class blocks and reverse class blocks. B/C reused the same finished-encoder embeddings; LoRA always used the original fine-tuned base and raw inputs after encoder-function changes. Brandâ€™s existing selected lookup and Platform were left untouched.

The fixed retention screen allowed at most 0.5 percentage-point accuracy loss, 0.01 macro-F1 loss, and 2 percentage-point recall loss for classes with at least 30 validation examples. All classes, including smaller/zero supports, remain in the detailed metrics. These are research diagnostics, not statistical noninferiority or production approval. LoRA was routed only where no simpler method also reduced proxy errors by at least 10% in **every** order; that routing threshold is not a demonstrated business need.

| Target | Maintenance updater | After accuracy, order range | After macro-F1, order range | CPU/GPU fit seconds, range | Retention passes |
|---|---|---:|---:|---:|---:|
| Segment | Existing head: softmax SGD | 96.9963â€“97.0131% | 0.945640â€“0.946059 | 0.032â€“0.045 | 3/3 |
| Segment | Existing head: PA-II | 81.4257â€“94.5114% | 0.552570â€“0.883756 | 0.073â€“0.097 | 0/3 |
| Segment | Existing head: AdamW | 96.9712â€“97.0298% | 0.944886â€“0.946703 | 0.615â€“2.518 | 2/3 |
| Segment | LoRA on existing fine-tuned base | 96.8373â€“97.0214% | 0.943431â€“0.946474 | 32.549â€“45.583 | 2/3 |
| Mnfr | Existing head: softmax SGD | 99.9661% | 0.995729 | 0.020â€“0.020 | 3/3 |
| Mnfr | Existing head: PA-II | 99.9831â€“99.9915% | 0.997847â€“0.998926 | 0.065â€“0.068 | 3/3 |
| Mnfr | Existing head: AdamW | 99.9661% | 0.995729 | 0.306â€“1.252 | 3/3 |
| Mnfr | LoRA on existing fine-tuned base | Not needed by routing rule | â€” | â€” | â€” |
| TargetAgeGroup | Existing head: softmax SGD | 99.7505% | 0.987426 | 0.023â€“0.025 | 3/3 |
| TargetAgeGroup | Existing head: PA-II | 94.8125â€“99.6817% | 0.611523â€“0.984018 | 0.072â€“0.073 | 1/3 |
| TargetAgeGroup | Existing head: AdamW | 99.7505â€“99.7677% | 0.987426â€“0.988026 | 0.391â€“1.902 | 3/3 |
| TargetAgeGroup | LoRA on existing fine-tuned base | 99.7677% | 0.988026â€“0.988049 | 35.403â€“36.834 | 3/3 |
| Sub-Segment | Existing head: softmax SGD | 88.5487â€“88.7044% | 0.659032â€“0.659893 | 0.034â€“0.035 | 0/3 |
| Sub-Segment | Existing head: PA-II | 61.4859â€“87.4762% | 0.513096â€“0.669827 | 0.070â€“0.072 | 0/3 |
| Sub-Segment | Existing head: AdamW | 88.4795â€“88.6957% | 0.654063â€“0.660255 | 0.327â€“1.140 | 0/3 |
| Sub-Segment | LoRA on existing fine-tuned base | 88.3671â€“88.5400% | 0.653412â€“0.653855 | 29.511â€“30.029 | 0/3 |

**Quality findings:** warm-start softmax SGD preserved Segmentâ€™s quality in every order, but fixed head updates did not resolve every hard proxy example. PA-II fixed all six Mnfr proxy errors and improved only two or three validation decisions; this is too little evidence for a general superiority claim. The same aggressive method caused major Segment/Sub-Segment regressions and severe order sensitivity. TargetAgeGroupâ€™s head-only updates and LoRA produced similarly tiny changes. Every Sub-Segment updater breached the fixed per-class retention safeguard, including LoRA; keep its current model unchanged. Frozen base weights do not mean the adapted function cannot forget.

| Target | One-time full historical/validation cache build | Measured 512-record preprocessing + embedding |
|---|---:|---:|
| Segment | 551.29 s | 0.696 + 0.972 s |
| Mnfr | 178.29 s | 0.407 + 0.965 s |
| TargetAgeGroup | 170.24 s | 0.595 + 0.890 s |
| Sub-Segment | 174.65 s | 0.416 + 0.884 s |

Embedding generation is separate from classifier updates. Full cache build includes research-only screening of the already-seen training population and fixed validation; it is not a recurring per-update cost. Cheap updaters were also reproduced in CPU-only workers without initializing CUDA, with exactly matching head deltas. Their standalone-worker RSS and sampled incremental RSS are recorded; LoRA resource peaks are reported at GPU-worker scope, not falsely attributed solely to adapter tensors. CPU timing includes ordinary warm-up noise and uses one timing observation per order, not a production latency benchmark.

Complete accuracy/forgetting/order comparison (`data/processed/maintenance_research_v1_20261114T060644Z/comparison.csv`, private local evidence), per-class precision/recall/F1 (`data/processed/maintenance_research_v1_20261114T060644Z/per_class_comparison.csv`, private local evidence), and costs including preprocessing, save/reload, validation, CPU/RSS and storage (`data/processed/maintenance_research_v1_20261114T060644Z/cost_comparison.csv`, private local evidence). Here, cold means uncached incoming vectors with the existing encoder resident, not a cold process start. Cycle costs include measured incoming encoding (or LoRA model/tokenization preparation), fitting, delta save/reload and fixed-population validation. Frozen-head validation reuses its verified cache; LoRA must recompute it. Review time, ingestion/UI overhead, electricity and dollar cost were not measured.

**Correction memory:** 512-record insert/fingerprint cycles took 0.039â€“0.064 seconds on CPU. Guarded exact-feature matching produced zero validation hits in all four targets, preserving baseline decisions; this held-out-group population does not establish real repeat-product coverage. UPC-only reuse introduced one new Segment error despite a valid checksum. Retailer/SKU mappings also had conflicts, and Sub-Segment had one conflicting full-feature fingerprint. Require trusted product identity **plus input/version checks**, provenance, revocation and conflict abstention; do not blindly reuse a prior label or pretend a feature fingerprint guarantees business identity. Actual human-confirmed corrections and traffic data are needed to measure benefit.

**Verification/resources:** all 45 final deltas were checked; validation accuracy, macro-F1 and per-class precision/recall were independently recomputed. CPU head replicas matched exactly; applicable optimizer states were checked. Saved LoRA deltas reconstructed the unchanged fine-tuned encoder, and seven positive checked-loader cases plus wrong-base/path-traversal tests passed. Final verification performed zero optimizer updates and initialized no CUDA. Every delta now carries an explicit original-checkpoint/label/input/payload binding; never infer a pretrained base from a bare adapter directory.

Shared elapsed through verification: **55.48 minutes / 360 allowed**. CPU: **0.845 core-hours**, including a conservative 40-second upper bound for one immediately stopped preliminary accounting-helper bug. Peak worker commit **3.757 GiB / 4 GiB**; peak sampled device usage **1219 MiB**, minimum sampled GPU free **4.664 GiB**. Four numerical threads/four-CPU affinity, one interop thread, a verified 3-GiB allocator cap, fresh one-second telemetry and 256-MiB monitor threshold were maintained. Device-wide 4.5-GiB-used/1.5-GiB-free values are sampled stop thresholds, not an instantaneous total-VRAM hard cap. No resource breach occurred. [PyTorch allocator scope](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html).

Preservation hashes matched original checkpoints/model artifacts, development files, tracked project files, original environment metadata/configuration and all 1,176 shared-overlay files. Tracked/staged diffs remained empty; only this report and the consolidated research directory are new. No dependencies were installed. The only technical retry was a missing declaration during resource-code consolidation, resolved before model work; its brief run was conservatively budgeted and all child jobs terminated.

**Recommended maintenance strategy:** keep the current models as the default. Add an append-only human-correction ledger and conservative exact-match suggestions, with mandatory review on every prediction. For genuine new labels, first trial a warm-start update to the current accepted head with cached frozen-encoder replay. Segmentâ€™s softmax SGD is a low-cost candidate; Mnfrâ€™s binary PA-II result warrants only a restricted pilot; TargetAgeGroup does not currently justify encoder adaptation; Sub-Segment needs label/conflict adjudication and better real correction evidence before any parameter update. Do not generalize a cheap method across targets merely because it is fast.

**Is LoRA necessary? Not demonstrated.** It was skipped for Mnfr, gave little extra value for TargetAgeGroup, and did not consistently improve Segment or preserve Sub-Segmentâ€™s class safeguards. Keep it as escalation only when a held-out, genuinely new confirmed cohort shows that frozen representations and safe head updates are insufficient. Replay is a useful inexpensive baseline ([primary research](https://arxiv.org/abs/1902.10486)); LoRA reduces trainable parameters ([original paper](https://arxiv.org/abs/2106.09685)), but neither establishes quality or prevents forgetting in this application by itself.

Later integration, requiring separate approval:

1. Record real confirmation events: target, trusted product key, full input fingerprint/version, reviewer/event provenance, original model/head version, accepted label and ontology version. Conflicts must be adjudicated, not silently overwritten.
2. Keep each targetâ€™s existing validation population closed to updates; deduplicate genuinely incoming groups against original training and evaluation membership. Do not use timestamps alone to claim an example is new. Preserve the protected test set.
3. Cache embeddings by encoder function (including active adapters), tokenizer/text contract and input fingerprint. A head-only change can reuse embeddings; a LoRA function change cannot. Start subsequent updates from the **latest accepted head**, with fixed historical-plus-confirmed replay and rollback to the original model.
4. Shadow-test fixed maintenance candidates against unchanged baselines on the original validation and a separately held-out incoming cohort. Track all per-class precision/recall, correct-to-wrong flips, order sensitivity and confidence reliability. Revisit calibration after accepted weight changes; do not add automatic approval.
5. Measure actual human correction rate and review time prospectively. Deploy no delta from this proxy study. Integrate only after explicit approval, using checked base/label/payload bindings and a tested rollback path.

Evidence and limits: one update cycle, seen-training error-enriched proxies, old labels without fresh adjudication, three synthetic within-cycle orders rather than chronological arrivals, reused development validation and no new-class/long-stream experiment. Proxy fit gains are not unseen-product accuracy. No original model was retrained from scratch, replaced or promoted.
