# Round 8D-A Five-Appliance Event Benchmark

This directory contains the small result bundle for the Round 8D-A Han-compatible five-real-appliance active-cycle/event-signature benchmark.

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

Start with `tables/model_event_metrics.csv`, `tables/h0_m6_reproduction_validation.csv`, `tables/model_size_estimate.csv`, `tables/shape_plus_improvement_summary.csv`, and the figures in `figures/`.
