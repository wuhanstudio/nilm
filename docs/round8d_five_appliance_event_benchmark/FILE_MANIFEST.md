# Benchmark File Manifest

This manifest lists the files included for the Round 8D-A five-appliance active-cycle event benchmark.

| File | Type | Required | Notes |
| --- | --- | --- | --- | --- |
| `main_redd_tm_five_appliance_shape_benchmark.py` | code | required | Self-contained report/smoke/H0-M6/model-size helper |
| `docs/round8d_five_appliance_event_benchmark/README.md` | documentation | required | Entry point for benchmark docs |
| `docs/round8d_five_appliance_event_benchmark/METHODOLOGY_AND_TERMS.md` | documentation | required | Project-specific terminology and step-by-step M6 encoding pipeline |
| `docs/round8d_five_appliance_event_benchmark/TECHNICAL_NOTE.md` | documentation | required | Protocol, feature, model, and caveat explanation |
| `docs/round8d_five_appliance_event_benchmark/REPRODUCTION_COMMANDS.md` | documentation | required | Manual commands for report, smoke, dry-run, H0-M6, and size estimate |
| `docs/round8d_five_appliance_event_benchmark/DEPLOYMENT_AND_PORTING_NOTES.md` | documentation | required | Tsetlin serialization, header compilation, and Arduino/Pico notes |
| `docs/round8d_five_appliance_event_benchmark/tables/model_event_metrics.csv` | result table | required | Main metric table |
| `docs/round8d_five_appliance_event_benchmark/tables/shape_plus_improvement_summary.csv` | result table | required | Shows F3 shape-plus improvement |
| `docs/round8d_five_appliance_event_benchmark/tables/h0_m6_reproduction_validation.csv` | validation table | required | Full H0/M6 reproduction observed-vs-archived metrics and runtime |
| `docs/round8d_five_appliance_event_benchmark/tables/model_size_estimate.csv` | estimate table | required | No-training M6 literal-position and state-size estimate |
| `docs/round8d_five_appliance_event_benchmark/tables/evidence_traceability_matrix.csv` | result table | required | Maps claims to evidence |
| `docs/round8d_five_appliance_event_benchmark/tables/feature_group_ablation_plan.csv` | planning table | optional | Documents future ablation plan |
| `docs/round8d_five_appliance_event_benchmark/figures/event_macro_f1_by_protocol_model.png` | figure | required | Macro F1 overview |
| `docs/round8d_five_appliance_event_benchmark/figures/per_appliance_f1_by_protocol.png` | figure | required | Per-appliance F1 overview |
| `docs/round8d_five_appliance_event_benchmark/figures/shape_plus_furnace_washer_f1.png` | figure | optional | Shape-plus furnace/washer focus |
| `docs/round8d_five_appliance_event_benchmark/figures/furnace_washer_confusion_focus.png` | figure | optional | Focused confusion audit |
| `docs/round8d_five_appliance_event_benchmark/figures/confusion_matrix_best_model_h0.png` | figure | required | H0 best confusion matrix |
| `docs/round8d_five_appliance_event_benchmark/figures/confusion_matrix_best_model_h1.png` | figure | required | H1 best confusion matrix |

The benchmark intentionally excludes raw REDD data, machine-specific paths, logs, output folders, model binaries, `.npy`, `.npz`, raw time-series snippets, and trained model artifacts.
