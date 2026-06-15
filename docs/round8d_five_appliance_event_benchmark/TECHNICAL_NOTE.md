# Technical Note: Five-Appliance Active-Cycle Event Benchmark

## What This Benchmark Is

Round 8D-A is a Han-compatible active-cycle / event-signature classification benchmark. It asks whether five real appliance event signatures can be separated when clean appliance-derived event windows are available.

## What This Benchmark Is Not

It is not deployable aggregate-main NILM performance. The headline H0/H1 results use appliance-derived matched transitions and event windows, so they should not be reported as aggregate-main sample-level NILM.

## Data And Episode Source

Episodes are generated from appliance channels using matched transitions. This makes the benchmark reference-only: appliance labels/channels define the event windows, while the classifier tests event-signature separability.

## Protocols

`H0_random_event_split` is a stratified random event split. It is closest to a Han-style event-signature demonstration and tests within-distribution event separability.

Because H0 randomly splits events, train and test can share house-specific appliance signatures. This is intentional for the optimistic Han-compatible reference benchmark, but it should not be interpreted as cross-house deployable NILM.

`H1_H3_heldout_reference` holds out House 3. It tests stricter cross-house transfer in the same reference-only event framing.

## Targets

The target appliances are fridge, microwave, dish washer, electric furnace, and washer dryer. `unknown` is excluded because it is an unmatched/background class, not a real appliance target.

## Feature Progression

`F0_han_core`: transition magnitudes and duration.

`F1_han_plus`: transition/duration features plus event-window summary statistics.

`F2_paper_style_8bit`: compact paper-style transition/duration features with train-standardized 8-bit booleanization.

`F3_event_shape_plus_reference`: transition features plus event-shape, internal-edge, subcycle proxy, active-fraction proxy, and pre/post context-derived descriptors.

## Feature Source And Deployability

| Feature group | Source in this benchmark | Deployability interpretation |
| --- | --- | --- |
| `F0_han_core` | appliance-derived matched event windows | reference-only |
| `F1_han_plus` | appliance-derived matched event windows plus window summaries | reference-only |
| `F2_paper_style_8bit` | appliance-derived matched event windows with train-only booleanization | reference-only |
| `F3_event_shape_plus_reference` | appliance-derived event windows with shape, internal edge, and pre/post context descriptors | reference-only |

F3 features are computed from appliance-derived event windows in this benchmark. They explain event-signature separability, not deployable aggregate-main NILM performance.

## Model Progression

`M0_simple_baseline`: nearest-centroid reference baseline.

`M1_Tsetlin_compact`: compact one-vs-rest TM with F0.

`M2_Tsetlin_paper_style`: paper-style one-vs-rest TM with F2.

`M3_Tsetlin_paper_style_balanced`: class-balanced paper-style TM.

`M5_shape_plus_simple_baseline`: simple baseline with F3.

`M6_Tsetlin_shape_plus_paper_style`: paper-style TM with F3; this is the final best H0/H1 model.

## Why The Final Jump Happened

The evidence supports a feature-set-level improvement from F3 shape features. H0 best macro F1 moved from about 0.7353 before shape-plus to about 0.9259 with M6. This should not be interpreted as proof of individual feature causality, because feature-group ablation has not yet been run.

## Appliance-Level Interpretation

Fridge is strong in both H0 and H1.

Microwave is strong in both H0 and H1.

Washer dryer is strong in both H0 and H1.

Dish washer is strong in H0 but weak in H1, so it remains a cross-house weakness.

Electric furnace improved strongly in H0 but remains weak in H1, so it also remains a cross-house weakness.

## Trade-Offs

The H0 score is high and useful as a supervisor-facing event-signature reference. The caveat is equally important: the event source is reference-only. H0 demonstrates separability; H1 diagnoses cross-house generalization. F3 shape features improve the benchmark but move beyond Han's original minimal transition/duration feature set.

## Unknowns

Feature-group ablation has not been run. Deployable aggregate-main NILM remains a separate route. The candidate upstream script now includes a self-contained H0/M6 reproduction path, but that long path still depends on a compatible local Tsetlin API.

Han's paper reports about `18 KB` for a simpler transition-duration TM setting. Round 8D-A M6 uses F3 shape-plus features, train-only 8-bit Booleanization, and five one-vs-rest models, so its expected footprint is larger. The candidate script includes a no-training `--mode model-size-estimate`, but exact MCU size requires future model export and compile.

Han's existing Tsetlin implementation can serialize trained objects to protobuf through `save_model()`, and `tsetlin_compile()` can generate inference headers from those protobufs. Because M6 is one-vs-rest, a deployable export would need five binary TM exports plus integration logic that chooses the appliance with the strongest binary evidence. These exports should be treated as optional generated artifacts, not default source files.

The inspected Arduino runtime includes a generic static-header inference example for Iris, but no Pico H / RP2040-specific example was found. A Pico-oriented Round 8D-A demonstrator would need an additional wrapper around five binary TM headers plus the F3 feature and Booleanization path.
