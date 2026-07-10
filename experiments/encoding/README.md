# Low-frequency Boolean encoders for NILM

This folder contains a reusable research implementation of the low-frequency
Boolean encoders used in the NILM + Tsetlin Machine experiments.

## Scope

The encoder input is a causal aggregate active-power window:

```text
main[t-W+1 : t]
```

The output is a Boolean vector suitable for a Tsetlin Machine.  Appliance
channels are not used by the encoder; they are labels/evaluation data only.

## Encoder modes

- `stats_bool`: compact multi-window statistical encoding.
- `stats_haar_bool`: `stats_bool` plus Haar-like block-difference features.
- `stats_haar_dct_bool`: `stats_haar_bool` plus low-order DCT-like coefficients.

Current project judgement:

- `stats_bool` is the first deployment candidate because it is compact and has
  the strongest current TM desktop pilot result.
- `stats_haar_bool` is useful as a low-frequency multiscale ablation.
- `stats_haar_dct_bool` is kept as a desktop ablation because it is heavier and
  has not yet shown a stable advantage.

## Reproduce the encoder smoke test

```bash
uv run python experiments/encoding/run_encoder_smoke.py --data-dir path/to/redd_csv
```

The smoke test loads `redd_house1_0.csv`, fits all three encoders on a subset of
aggregate windows, checks output shapes, and verifies batch-vs-streaming
Boolean equivalence.

## Notes

This is a v0.1 research module.  Default parameters may still change after
additional TM and Arduino experiments.  The embedded C/C++ implementation is not
included here yet.
