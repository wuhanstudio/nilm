# Reproduction Commands

## Data Setup

For data-dependent modes, set an environment variable or pass `--data-dir`.

```bash
export NILM_REDD_DATA_DIR=/path/to/converted/redd
```

Do not commit machine-specific data paths to docs, scripts, tables, or figures.

Expected converted REDD schema:

* Files match `redd_house*_*.csv`.
* Required appliance columns for H0/M6: `fridge`, `microwave`, `dish washer`, `electric furnace`, `washer dryer`.
* The H0/M6 path uses appliance-derived event windows and is reference-only.

## Report Mode

No REDD data required:

```bash
python files/main_redd_tm_five_appliance_shape_benchmark.py --mode report
```

## Smoke Mode

Requires converted REDD CSVs:

```bash
python files/main_redd_tm_five_appliance_shape_benchmark.py --mode smoke
```

This is a tiny sanity path, not a headline reproduction.
It may not include all five appliances because it intentionally reads only a few small CSV chunks.

## H0/M6 Long Reproduction

Dry-run/workload validation, no TM training:

```bash
python files/main_redd_tm_five_appliance_shape_benchmark.py --mode h0-m6 \
  --data-dir "$NILM_REDD_DATA_DIR" \
  --dry-run
```

This validates REDD file discovery, required columns, target appliance list, episode generation workload, feature columns, split construction, and expected model settings.

Full H0/M6 reproduction:

```bash
python files/main_redd_tm_five_appliance_shape_benchmark.py --mode h0-m6 \
  --data-dir "$NILM_REDD_DATA_DIR" \
  --output-dir /tmp/round8d_h0_m6
```

This mode implements the H0/M6 path:

* appliance-derived P2 nearest-opposite pairing with `+50 W` and `-50 W` transition thresholds
* maximum episode duration `20000` samples
* `F3_event_shape_plus_reference` features
* H0 stratified random event split with seed `6072` and test fraction `0.30`
* train-only standardized 8-bit booleanization
* one-vs-rest Tsetlin Machine, `n_clause=200`, `n_state=50`, `T=20`, `s=6.0`, `epochs=20`

Expected archived H0/M6 metrics:

| Metric | Expected |
| --- | ---: |
| accuracy | 0.9411 |
| macro F1 | 0.9259 |
| fridge F1 | 0.9663 |
| microwave F1 | 0.9909 |
| dish washer F1 | 0.8408 |
| electric furnace F1 | 0.8900 |
| washer dryer F1 | 0.9415 |

Observed H0/M6 reproduction:

| Metric | Observed |
| --- | ---: |
| accuracy | 0.9492 |
| macro precision | 0.9620 |
| macro recall | 0.9195 |
| macro F1 | 0.9378 |
| fridge F1 | 0.9660 |
| microwave F1 | 0.9909 |
| dish washer F1 | 0.8662 |
| electric furnace F1 | 0.9129 |
| washer dryer F1 | 0.9527 |
| runtime | about 30m28s |

Allow small tolerance for local Tsetlin implementation and stochastic update-order differences. A deviation larger than a few F1 points should be investigated before using the reproduced result.

Runtime warning: this mode trains five one-vs-rest TMs for 20 epochs and may take significant time.

If Han's Tsetlin API is unavailable or incompatible, the script should stop with a clear `Tsetlin implementation not found or incompatible` message.

Validation status: report mode, smoke mode, H0/M6 dry-run, and one full H0/M6 long reproduction were tested. The observed full H0/M6 metrics were comparable to archived metrics and slightly higher in that run; this is not a bit-exact reproduction claim. The same commands are intended to be manually runnable in PowerShell or shell with a uv-managed Python environment.

## Full Reproduction Warning

Full multi-model Round 8D reproduction is intentionally omitted from this script. The included reproduction path focuses on the no-data report, smoke sanity path, and the H0/M6 headline model.

## Model-Size Estimate

No REDD data and no training required:

```bash
python files/main_redd_tm_five_appliance_shape_benchmark.py --mode model-size-estimate
```

The estimate uses:

* continuous feature count from F3
* `8` bits per feature
* literal count `2 * boolean_features`
* `200` clauses
* `5` one-vs-rest models
* `n_state = 50`

Han's paper reports about `18 KB` for the simpler transition-duration setting. Round 8D-A M6 is expected to be larger because it uses F3 shape-plus features and five one-vs-rest models. Exact MCU footprint requires future export/compile.

## Optional Model Export

Model export is disabled by default. To explicitly export inference protobuf/header artifacts during a full H0/M6 run:

```bash
python files/main_redd_tm_five_appliance_shape_benchmark.py --mode h0-m6 \
  --data-dir "$NILM_REDD_DATA_DIR" \
  --output-dir /tmp/round8d_h0_m6 \
  --export-model-dir /tmp/round8d_h0_m6_models
```

This would export five one-vs-rest binary TMs. Expected generated files include:

* `m6_ovr_<appliance>_inference.pb`
* `m6_ovr_<appliance>_inference.h`
* shared `tsetlin_model.h`

Do not commit these generated model artifacts by default. They are optional local outputs for footprint/export investigation.
