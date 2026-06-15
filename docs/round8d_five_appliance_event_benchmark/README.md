# Round 8D-A Five-Appliance Event Benchmark

This directory contains the result bundle for the Round 8D-A Han-compatible five-real-appliance active-cycle/event-signature benchmark.

Headline H0 best model:

* `M6_Tsetlin_shape_plus_paper_style`
* Accuracy about `0.9411`
* Macro F1 about `0.9259`

Full H0/M6 reproduction was executed successfully from the candidate script. Observed metrics were comparable to the archived metrics and slightly higher in that run: accuracy `0.9492`, macro F1 `0.9378`, runtime `1827.7s`. This is not a bit-exact claim.

Headline H1 best model:

* `M6_Tsetlin_shape_plus_paper_style`
* Accuracy about `0.7962`
* Macro F1 about `0.6514`

Important caveat: this is reference-only. H0/H1 use appliance-derived matched transitions and event windows, so these scores are not deployable aggregate-main NILM performance.

The five real appliance targets are fridge, microwave, dish washer, electric furnace, and washer dryer. H0 is the main Han-compatible reference protocol because it uses a stratified random event split, closest to an active-cycle signature demonstration. H1 holds out House 3 and is included as a stricter cross-house diagnostic.

Key terms and method: `METHODOLOGY_AND_TERMS.md` defines the project-specific protocol names, P2 pairing, F3 shape-plus features, train-only 8-bit Booleanization, and the five one-vs-rest TM design.

Read:

* `METHODOLOGY_AND_TERMS.md` for definitions and the step-by-step M6 encoding pipeline.
* `TECHNICAL_NOTE.md` for protocol, feature, model, and reference-only details.
* `REPRODUCTION_COMMANDS.md` for report, smoke, dry-run, H0/M6, and model-size commands.
* `DEPLOYMENT_AND_PORTING_NOTES.md` for protobuf/header export and Arduino/Pico notes.
* `FILE_MANIFEST.md` for the committed file list.

Start with `tables/model_event_metrics.csv`, `tables/h0_m6_reproduction_validation.csv`, `tables/model_size_estimate.csv`, `tables/shape_plus_improvement_summary.csv`, and the figures in `figures/`.
