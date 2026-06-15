# Deployment and Porting Notes

## Self-Contained Benchmark Script

The benchmark script `main_redd_tm_five_appliance_shape_benchmark.py` is self-contained for:

* `--mode report`: reads bundled result tables only.
* `--mode smoke`: runs a tiny standalone sanity path if converted REDD CSVs are available.
* `--mode h0-m6`: discovers converted REDD CSVs, builds appliance-derived P2 episodes, computes F3 shape-plus features, applies train-only standardized 8-bit booleanization, trains one-vs-rest TMs, and writes compact metrics if an output directory is supplied.
* `--mode model-size-estimate`: prints a no-training M6 footprint estimate from feature count, Booleanization width, literal count, clause count, OVR model count, and `n_state`.
* `--export-model-dir <path>`: optional explicit H0/M6 export hook. It is disabled by default and only has an effect during a full `--mode h0-m6` run.

It does not require any non-repository benchmark helper modules.

## Components Now Ported

The script contains:

* REDD converted CSV discovery compatible with Han's repo layout.
* Appliance-derived P2 nearest-opposite transition pairing.
* Round 8D `F3_event_shape_plus_reference` feature extraction.
* Train-standardized 8-bit booleanization.
* One-vs-rest Tsetlin Machine wrapper for multiclass event classification.
* Metric and confusion-matrix writers.

## Dependency Status

Smoke mode is intentionally not exact Round 8D reproduction; it is only a sanity check that REDD columns can be read and appliance-derived event rows can be generated. H0/M6 mode requires `numpy`, `pandas`, and a compatible Han `Tsetlin` class with `step()` and `predict()` support.

If Han's local Tsetlin API differs, H0/M6 mode should fail with a clear `Tsetlin implementation not found or incompatible` message rather than silently using a different classifier.

## Remaining Limitations

* The long H0/M6 mode was implemented and completed once.
* H0/M6 dry-run/workload validation passed.
* Observed H0/M6 runtime was about `30m28s`, with accuracy `0.9492` and macro F1 `0.9378`.
* No trained TM model artifacts were saved by the candidate script. The trained one-vs-rest models existed only in Python process memory.
* Re-run full H0/M6 if environment-specific runtime numbers are desired.
* Han's paper reports about `18 KB` for the simpler transition-duration setting. Round 8D-A M6 is expected to be larger; exact MCU footprint requires future model export and compile.

## Model Export Investigation

Han's current Tsetlin path supports native protobuf serialization:

* `tsetlin/tsetlin.py` defines `Tsetlin.save_model(path, type="training"|"inference")`.
* `type="inference"` writes compressed inference clauses into a `tsetlin_pb2.Tsetlin` protobuf.
* `type="training"` writes full trainable clause state.
* `tsetlin/tsetlin_pb2.py` defines the protobuf messages `Tsetlin`, `Clause`, and `ClauseCompressed`.

The Round 8D-A M6 design is one-vs-rest, so exporting it requires five binary TMs:

* `m6_ovr_fridge_inference.pb`
* `m6_ovr_microwave_inference.pb`
* `m6_ovr_dish_washer_inference.pb`
* `m6_ovr_electric_furnace_inference.pb`
* `m6_ovr_washer_dryer_inference.pb`

`tsetlin/compiler/write.py` provides `tsetlin_compile(model_path, out_path)`. It reads a protobuf model, emits a shared `tsetlin_model.h`, emits the requested inference header such as `m6_ovr_fridge_inference.h`, and prints the compiled size. Because `tsetlin_compile()` writes `tsetlin_model.h` relative to the current working directory, the candidate script changes into the explicit export directory before compiling headers.

Exported protobuf/header files are generated artifacts. They should not be committed by default. Keep them as optional local outputs unless upstream maintainers explicitly request checked-in reference exports.

## Arduino / Pico Integration Notes

Han's repo already includes a generic Arduino static-header inference example at `arduino/lime-tm/examples/iris_inference/`. That example includes a generated model header, Booleanizes inputs, calls `tsetlin_evaluate()`, and reads class votes from the Arduino runtime in `arduino/lime-tm/src/`.

No Pico H / RP2040-specific example was found in the inspected tree. The existing Arduino runtime appears board-generic, with AVR-specific `PROGMEM` branches and ordinary static-header paths for non-AVR Arduino builds.

Round 8D-A M6 is not a single five-class TM. It is five one-vs-rest binary TMs. An Arduino/Pico wrapper would therefore need to:

1. Include or otherwise reference five generated binary TM headers.
2. Compute the same F3 continuous features and train-standardized 8-bit Booleanization used by the Python reproduction path.
3. Call `tsetlin_evaluate()` once per binary appliance model.
4. Compare each model's positive-vs-negative vote margin.
5. Select the appliance with the strongest positive evidence, or optionally return an unknown/no-event state if all positive margins are weak.

This wrapper is not implemented in the current benchmark script. It should be a future deployment/export task after review of the benchmark reproduction path.

Additional limitations:

* Exact numeric agreement may depend on Han's local Tsetlin implementation details and random update order.
* `--mode full` is intentionally omitted until the single headline H0/M6 path is reviewed.
* The benchmark remains reference-only because H0/H1 use appliance-derived event windows.

## Recommended Review Path

1. Review the reference-only caveat.
2. Run `--mode report`.
3. Run `--mode smoke` against converted REDD CSVs.
4. Run `--mode h0-m6` after confirming Han's local Tsetlin API is compatible.

Avoid claiming deployable aggregate-main NILM performance from this reference benchmark.
