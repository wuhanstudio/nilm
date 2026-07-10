"""Causal low-frequency Boolean encoders for NILM aggregate windows.

The encoders in this module convert a causal aggregate active-power window
``main[t-W+1:t]`` into Boolean features for Tsetlin Machine input.  They are
written as a small reusable research module, separate from the larger Round 6B
and Round 6C experiment scripts.

Design constraints:

- Use aggregate ``main`` only.  Appliance channels are labels/evaluation data,
  not encoder inputs.
- Be causal.  Runtime code must not look into future samples.
- Fit Boolean thresholds on training windows only.
- Keep batch and streaming transforms equivalent.
- Keep the compact encoder Arduino-friendly and reserve heavier features for
  ablation.

Available modes:

``stats_bool``
    Multi-window statistical summaries, then quantile Booleanization.

``stats_haar_bool``
    ``stats_bool`` plus Haar-like block-difference features that describe
    low-frequency multi-scale shape changes.

``stats_haar_dct_bool``
    ``stats_haar_bool`` plus a small number of low-order DCT-like coefficients.
    This is currently intended as a desktop ablation rather than the first
    Arduino target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

EncoderMode = Literal["stats_bool", "stats_haar_bool", "stats_haar_dct_bool"]


@dataclass(frozen=True)
class EncoderCostEstimate:
    """Small deployment-oriented summary for one encoder configuration."""

    mode: str
    window_length: int
    continuous_features: int
    boolean_features: int
    thresholds: int
    buffer_floats: int
    rough_ops_per_window: int
    arduino_suitability: str


class LowFreqBooleanEncoder:
    """Train-fitted Boolean encoder for causal aggregate-power windows.

    Parameters
    ----------
    mode:
        One of ``stats_bool``, ``stats_haar_bool`` or ``stats_haar_dct_bool``.
    window_length:
        Number of samples in the causal window.  Runtime input is interpreted as
        ``main[t-window_length+1:t]``.
    subwindows:
        Tail subwindows used for multi-resolution features.  Each value must be
        less than or equal to ``window_length``.
    quantiles:
        Quantile thresholds used for thermometer-style Booleanization.  Each
        continuous feature produces one bit for each quantile: ``feature > q``.
    dct_bins:
        Number of low-order DCT-like coefficients per subwindow.  Only used by
        ``stats_haar_dct_bool``.
    eps:
        Small numerical constant used to avoid division-by-zero style edge cases.

    Notes
    -----
    ``fit`` only computes feature thresholds.  It does not use labels.  Call
    ``transform`` on any aggregate-window matrix after fitting.  Call
    ``transform_streaming`` on a raw aggregate stream to reproduce batch results
    with a sliding causal buffer.
    """

    def __init__(
        self,
        mode: EncoderMode = "stats_bool",
        window_length: int = 128,
        subwindows: Sequence[int] = (16, 32, 64, 128),
        quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
        dct_bins: int = 6,
        eps: float = 1e-9,
    ) -> None:
        if mode not in {"stats_bool", "stats_haar_bool", "stats_haar_dct_bool"}:
            raise ValueError(f"Unknown encoder mode: {mode!r}")
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        if not subwindows:
            raise ValueError("subwindows must not be empty")
        if max(subwindows) > window_length:
            raise ValueError("all subwindows must be <= window_length")
        if min(subwindows) < 2:
            raise ValueError("all subwindows must be at least 2 samples")
        if any(q <= 0.0 or q >= 1.0 for q in quantiles):
            raise ValueError("quantiles must be between 0 and 1")
        if dct_bins < 0:
            raise ValueError("dct_bins must be non-negative")

        self.mode = mode
        self.window_length = int(window_length)
        self.subwindows = tuple(int(x) for x in subwindows)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.dct_bins = int(dct_bins)
        self.eps = float(eps)

        self.feature_names_: list[str] | None = None
        self.thresholds_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.thresholds_ is not None

    def fit(self, windows: np.ndarray) -> "LowFreqBooleanEncoder":
        """Fit Boolean thresholds on training aggregate windows only."""

        features, names = self._continuous_features(windows)
        self.feature_names_ = names
        self.thresholds_ = np.quantile(features, self.quantiles, axis=0).T
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Encode aggregate windows into Boolean features.

        Returns
        -------
        np.ndarray
            Boolean matrix with shape ``[n_windows, n_cont_features * n_quantiles]``.
        """

        self._require_fitted()
        features, names = self._continuous_features(windows)
        if names != self.feature_names_:
            raise RuntimeError("feature definition changed after fit")
        return self._booleanize(features)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        """Fit thresholds and transform the same training windows."""

        return self.fit(windows).transform(windows)

    def transform_streaming(self, main_stream: Sequence[float]) -> np.ndarray:
        """Encode a raw aggregate stream with a causal sliding window.

        The first output row corresponds to anchor index ``window_length - 1``.
        This method intentionally uses the same continuous-feature path as
        ``transform`` so that batch and streaming outputs can be compared exactly
        in Python before implementing an embedded version.
        """

        stream = np.asarray(main_stream, dtype=np.float64)
        if stream.ndim != 1:
            raise ValueError("main_stream must be one-dimensional")
        if len(stream) < self.window_length:
            return np.zeros((0, self.n_boolean_features()), dtype=np.uint8)

        windows = sliding_windows(stream, self.window_length)
        return self.transform(windows)

    def n_continuous_features(self) -> int:
        names = self.feature_definitions()
        return len(names)

    def n_boolean_features(self) -> int:
        return self.n_continuous_features() * len(self.quantiles)

    def feature_definitions(self) -> list[str]:
        """Return feature names without fitting thresholds."""

        dummy = np.zeros((1, self.window_length), dtype=np.float64)
        _, names = self._continuous_features(dummy)
        return names

    def boolean_feature_names(self) -> list[str]:
        """Return names for thresholded Boolean features."""

        names = self.feature_definitions()
        out: list[str] = []
        for name in names:
            for q in self.quantiles:
                out.append(f"{name}>q{q:g}")
        return out

    def thresholds_dataframe(self) -> pd.DataFrame:
        """Return fitted thresholds as a tidy DataFrame."""

        self._require_fitted()
        rows = []
        assert self.feature_names_ is not None
        assert self.thresholds_ is not None
        for feature_idx, name in enumerate(self.feature_names_):
            for q_idx, q in enumerate(self.quantiles):
                rows.append(
                    {
                        "mode": self.mode,
                        "feature": name,
                        "quantile": q,
                        "threshold": float(self.thresholds_[feature_idx, q_idx]),
                    }
                )
        return pd.DataFrame(rows)

    def feature_definitions_dataframe(self) -> pd.DataFrame:
        """Return feature definitions in a CSV-friendly form."""

        rows = []
        for idx, name in enumerate(self.feature_definitions()):
            rows.append({"mode": self.mode, "feature_index": idx, "feature": name})
        return pd.DataFrame(rows)

    def cost_estimate(self) -> EncoderCostEstimate:
        """Return a rough operation and memory estimate for deployment planning."""

        cont = self.n_continuous_features()
        bools = self.n_boolean_features()

        # These estimates are deliberately rough.  They are meant to compare
        # encoder sizes, not to replace a real embedded profiler.
        stats_ops = len(self.subwindows) * 120
        haar_ops = len(self.subwindows) * 140 if "haar" in self.mode else 0
        dct_ops = len(self.subwindows) * self.dct_bins * self.window_length if "dct" in self.mode else 0
        rough_ops = stats_ops + haar_ops + dct_ops

        if self.mode == "stats_bool":
            suitability = "high"
        elif self.mode == "stats_haar_bool":
            suitability = "medium"
        else:
            suitability = "low-medium"

        return EncoderCostEstimate(
            mode=self.mode,
            window_length=self.window_length,
            continuous_features=cont,
            boolean_features=bools,
            thresholds=bools,
            buffer_floats=self.window_length,
            rough_ops_per_window=rough_ops,
            arduino_suitability=suitability,
        )

    # ------------------------------------------------------------------
    # Internal feature extraction
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self.thresholds_ is None:
            raise RuntimeError("encoder is not fitted; call fit() first")

    def _booleanize(self, features: np.ndarray) -> np.ndarray:
        assert self.thresholds_ is not None
        bits = features[:, :, None] > self.thresholds_[None, :, :]
        return bits.reshape(features.shape[0], -1).astype(np.uint8)

    def _continuous_features(self, windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
        windows = np.asarray(windows, dtype=np.float64)
        if windows.ndim != 2:
            raise ValueError("windows must have shape [n_windows, window_length]")
        if windows.shape[1] != self.window_length:
            raise ValueError(
                f"expected window_length {self.window_length}, got {windows.shape[1]}"
            )

        pieces: list[np.ndarray] = []
        names: list[str] = []

        stats, stat_names = self._stats_features(windows)
        pieces.append(stats)
        names.extend(stat_names)

        if self.mode in {"stats_haar_bool", "stats_haar_dct_bool"}:
            haar, haar_names = self._haar_features(windows)
            pieces.append(haar)
            names.extend(haar_names)

        if self.mode == "stats_haar_dct_bool":
            dct, dct_names = self._dct_features(windows)
            pieces.append(dct)
            names.extend(dct_names)

        return np.concatenate(pieces, axis=1), names

    def _stats_features(self, windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
        cols: list[np.ndarray] = []
        names: list[str] = []

        for length in self.subwindows:
            x = windows[:, -length:]
            diffs = np.diff(x, axis=1)
            prefix = f"stats_w{length}"

            feature_map = {
                "last": x[:, -1],
                "mean": x.mean(axis=1),
                "std": x.std(axis=1),
                "min": x.min(axis=1),
                "max": x.max(axis=1),
                "range": x.max(axis=1) - x.min(axis=1),
                "last_minus_first": x[:, -1] - x[:, 0],
                "last_minus_mean": x[:, -1] - x.mean(axis=1),
                "mean_abs_diff": np.abs(diffs).mean(axis=1),
                "max_abs_diff": np.abs(diffs).max(axis=1),
            }

            for short_name, values in feature_map.items():
                cols.append(values)
                names.append(f"{prefix}_{short_name}")

        return np.column_stack(cols), names

    def _haar_features(self, windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
        cols: list[np.ndarray] = []
        names: list[str] = []

        for length in self.subwindows:
            x = windows[:, -length:]
            prefix = f"haar_w{length}"

            # Adjacent block mean differences at 2, 4 and 8 blocks.
            # This is a cheap Haar-like approximation: sums/means and
            # differences only, suitable as a Python reference for later C code.
            for n_blocks in (2, 4, 8):
                if length % n_blocks != 0:
                    continue
                block_len = length // n_blocks
                blocks = x.reshape(x.shape[0], n_blocks, block_len).mean(axis=2)
                adjacent = np.diff(blocks, axis=1)
                for idx in range(adjacent.shape[1]):
                    cols.append(adjacent[:, idx])
                    names.append(f"{prefix}_b{n_blocks}_adjdiff{idx}")

            # A few non-adjacent low-frequency contrasts.  These help represent
            # broad shape changes without requiring a full wavelet transform.
            half = length // 2
            quarter = length // 4
            first_half = x[:, :half].mean(axis=1)
            second_half = x[:, half:].mean(axis=1)
            first_quarter = x[:, :quarter].mean(axis=1)
            last_quarter = x[:, -quarter:].mean(axis=1)
            middle_half = x[:, quarter:-quarter].mean(axis=1) if quarter > 0 else x.mean(axis=1)

            extras = {
                "first_half_minus_second_half": first_half - second_half,
                "first_quarter_minus_last_quarter": first_quarter - last_quarter,
                "middle_half_minus_edges": middle_half - 0.5 * (first_quarter + last_quarter),
            }
            for short_name, values in extras.items():
                cols.append(values)
                names.append(f"{prefix}_{short_name}")

        return np.column_stack(cols), names

    def _dct_features(self, windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
        cols: list[np.ndarray] = []
        names: list[str] = []

        for length in self.subwindows:
            x = windows[:, -length:]
            centered = x - x.mean(axis=1, keepdims=True)
            basis = dct_basis(length, self.dct_bins)
            coeffs = centered @ basis.T
            for k in range(coeffs.shape[1]):
                cols.append(coeffs[:, k])
                names.append(f"dct_w{length}_k{k + 1}")

        return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def sliding_windows(stream: Sequence[float], window_length: int) -> np.ndarray:
    """Return causal sliding windows over a one-dimensional stream.

    The output row ``i`` corresponds to stream anchor ``i + window_length - 1``.
    """

    x = np.asarray(stream, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("stream must be one-dimensional")
    if len(x) < window_length:
        return np.zeros((0, window_length), dtype=np.float64)

    # sliding_window_view returns a view when possible and keeps this helper
    # concise.  The returned matrix is safe to pass into feature extraction.
    return np.lib.stride_tricks.sliding_window_view(x, window_length)


def dct_basis(length: int, n_bins: int) -> np.ndarray:
    """Return a small orthonormal DCT-II basis for bins 1..n_bins.

    Bin 0 is the DC component and is intentionally excluded because window mean
    is already included in ``stats_bool``.  The implementation avoids requiring
    SciPy inside the reusable encoder module.
    """

    if n_bins <= 0:
        return np.zeros((0, length), dtype=np.float64)

    n = np.arange(length, dtype=np.float64)
    basis = []
    max_k = min(n_bins, length - 1)
    for k in range(1, max_k + 1):
        vec = np.cos(np.pi * (n + 0.5) * k / length)
        vec *= np.sqrt(2.0 / length)
        basis.append(vec)
    return np.vstack(basis)


def cost_estimates_dataframe(encoders: Iterable[LowFreqBooleanEncoder]) -> pd.DataFrame:
    """Collect deployment cost estimates for multiple encoders."""

    rows = [encoder.cost_estimate().__dict__ for encoder in encoders]
    return pd.DataFrame(rows)


def save_encoder_artifacts(encoder: LowFreqBooleanEncoder, output_dir: str | Path) -> None:
    """Save feature definitions, thresholds and cost estimate for review."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    encoder.feature_definitions_dataframe().to_csv(out / f"{encoder.mode}_feature_definitions.csv", index=False)
    if encoder.is_fitted:
        encoder.thresholds_dataframe().to_csv(out / f"{encoder.mode}_thresholds.csv", index=False)
    pd.DataFrame([encoder.cost_estimate().__dict__]).to_csv(out / f"{encoder.mode}_cost_estimate.csv", index=False)
