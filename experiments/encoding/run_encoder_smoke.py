#!/usr/bin/env python3
"""Small smoke test for the reusable low-frequency Boolean encoder module.

This script is intentionally lightweight.  It does not train a model and does
not require appliance labels.  It simply loads one REDD House 1 aggregate stream,
fits each encoder on a subset of causal windows, verifies transform shapes, and
checks batch-vs-streaming equivalence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lowfreq_boolean_encoder import LowFreqBooleanEncoder, cost_estimates_dataframe, sliding_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/redd_csv"))
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--max-fit-windows", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.data_dir / f"redd_house1_{args.chunk}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    main_stream = df["main"].astype(float).to_numpy()
    windows = sliding_windows(main_stream, args.window_length)
    fit_windows = windows[: args.max_fit_windows]

    encoders = [
        LowFreqBooleanEncoder("stats_bool", window_length=args.window_length),
        LowFreqBooleanEncoder("stats_haar_bool", window_length=args.window_length),
        LowFreqBooleanEncoder("stats_haar_dct_bool", window_length=args.window_length),
    ]

    rows = []
    for encoder in encoders:
        encoder.fit(fit_windows)
        batch = encoder.transform(fit_windows)
        streaming = encoder.transform_streaming(main_stream[: args.max_fit_windows + args.window_length - 1])
        streaming = streaming[: len(batch)]
        mismatch = float(np.mean(batch != streaming)) if batch.size else 0.0
        rows.append(
            {
                "mode": encoder.mode,
                "continuous_features": encoder.n_continuous_features(),
                "boolean_features": encoder.n_boolean_features(),
                "batch_shape": str(tuple(batch.shape)),
                "streaming_shape": str(tuple(streaming.shape)),
                "streaming_mismatch_rate": mismatch,
            }
        )

    print("=== Encoder smoke summary ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== Cost estimates ===")
    print(cost_estimates_dataframe(encoders).to_string(index=False))


if __name__ == "__main__":
    main()
