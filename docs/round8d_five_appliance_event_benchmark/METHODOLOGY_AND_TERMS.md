# Methodology and Terms

This note defines the project-specific terms used in the Round 8D-A five-appliance active-cycle event benchmark.

## Terminology

`H0_random_event_split`: The main Han-compatible reference protocol. Appliance-derived event examples are split randomly with class stratification using seed `6072` and test fraction `0.30`. H0 measures within-distribution event-signature separability.

`H1_H3_heldout_reference`: A stricter diagnostic protocol that holds out House 3. H1 measures cross-house transfer under the same reference-only event framing.

`active-cycle event`: A finite appliance activity interval bounded by a rising transition and a later falling transition.

`appliance-derived matched event window`: An event window generated from an appliance channel rather than from aggregate mains. It gives a clean reference event for signature classification.

`reference-only benchmark`: A benchmark that uses appliance channels to define event windows. It is useful for event-signature analysis, but it is not a deployable aggregate-main NILM result.

`P2 nearest-opposite pairing`: The event pairing rule used for H0/M6 reproduction. A rising edge of at least `+50 W` is paired with the nearest later unused falling edge of at most `-50 W`, with maximum duration `20000` samples.

`F3 shape-plus features`: The final event feature set. It includes transition magnitudes and duration plus event-window shape summaries, internal edge counts, subcycle proxy, active-fraction proxy, and pre/post context descriptors.

`M6_Tsetlin_shape_plus_paper_style`: The final Round 8D-A Tsetlin Machine model. It uses F3 shape-plus features, train-only 8-bit Booleanization, and five one-vs-rest binary TMs.

`train-only 8-bit Booleanization`: The training split alone determines feature mean, standard deviation, and scaling bounds. Each standardized continuous feature is mapped into one byte and unpacked into eight Boolean bits. Test data is transformed with the training transform.

`one-vs-rest / OVR TM`: A multiclass design built from one binary TM per appliance. Each binary TM scores whether an event belongs to one appliance or the rest.

`H0/M6 reproduction validation`: The full H0/M6 reproduction run completed from the benchmark script. Observed metrics were comparable to the archived metrics and slightly higher in that run; this is not a bit-exact claim.

`model-size estimate`: A no-training estimate based on continuous feature count, 8-bit Booleanization, literal count, clause count, OVR model count, and `n_state`. Exact MCU footprint requires export and compile.

## Relationship to Han's Original REDD TM Route

Han's original route uses matched transitions to form active-cycle style examples. The original core inputs are transition and duration features. Those continuous features are Booleanized before TM training, and the original classifier is a multiclass TM.

This benchmark keeps the active-cycle/event-signature direction and extends it to five real REDD appliances. It adds richer event-shape features, explicit H0/H1 protocols, full H0/M6 reproduction validation, and a model-size estimate. This is an extension of the original route, not a replacement.

## Round 8D-A M6 Encoding Pipeline

1. Discover converted REDD CSV files.
2. Load the target appliance columns: fridge, microwave, dish washer, electric furnace, and washer dryer.
3. Build appliance-derived reference episodes.
4. Apply P2 nearest-opposite pairing: `+50 W` rising edge, nearest later `-50 W` falling edge, maximum duration `20000` samples.
5. Extract F3 shape-plus features from each event window.
6. Split H0 with stratified random event split, seed `6072`, test fraction `0.30`.
7. Fit Booleanization only on training features.
8. Standardize features with training mean/std, divide by `3 * std`, clip to `[-1, 1]`, map to `[0, 255]`, then unpack each byte into eight Boolean bits.
9. Train five binary one-vs-rest TMs.
10. Score each test event with the five binary models and choose the appliance with the largest positive-vs-negative vote margin.

## Improvements Over the Original Route

The benchmark expands to five real appliances instead of the smaller original setting. Washer dryer is included as a real appliance rather than using `unknown` as a fifth class.

F3 shape-plus features capture event profile information beyond transition and duration. Train-only Booleanization avoids test leakage. H0 and H1 separate within-distribution event separability from cross-house transfer diagnostics. The package also includes full H0/M6 reproduction validation and a model-size estimate.

## Limitations

This is still reference-only. It uses appliance-derived event windows and should not be claimed as deployable aggregate-main NILM performance.

M6 is larger than the simpler transition-duration setting. Pico deployment requires protobuf/header export and a wrapper that runs five one-vs-rest binary TMs.
