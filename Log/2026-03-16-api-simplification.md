# TransFit API Simplification Log (2026-03-16)

## Goal
- Keep the public fitting API explicit.
- Reduce the need for standard users to learn `Context` and `Distance`.
- Preserve compatibility for advanced workflows and forward-model usage.

## Changes
1. Kept explicit fitting entry points
- `transfit/api.py`
  - Removed the generic `fit(...)` wrapper from the advertised public API.
  - Kept `fit_bol(...)` and `fit_multiband(...)` as the recommended public entry points.

2. Simplified inputs for explicit fit functions
- `transfit/api.py`
  - `fit_bol(...)` now accepts direct context-like inputs:
    - `z`
    - `DL_cm`
  - `fit_multiband(...)` now accepts direct context-like inputs:
    - `z`
    - `DL_cm`
    - `filters`
    - optional `y_kind`
- Result:
  - Public fitting no longer exposes `ctx`.
  - Users stay with explicit `fit_bol` / `fit_multiband` APIs without being forced to build `Context` manually.

3. Tightened the advertised top-level namespace
- `transfit/__init__.py`
  - Removed `fit` from top-level `__all__`.
  - Kept top-level emphasis on:
    - `BolometricData`
    - `MultiBandData`
    - `fit_bol`
    - `fit_multiband`
    - `save`
    - `load`
    - `default_outpath`
    - `plot`

4. Rewrote README to match the explicit-fit workflow
- `readme.md`
  - Changed the standard workflow to:
    - `tf.fit_bol(...)`
    - `tf.fit_multiband(...)`
  - Clarified that `Context` is mainly for advanced and forward-model workflows.
  - Updated the minimal example to use `fit_bol(...)` and `plot.fit_bol(...)`.

5. Synced the tutorial notebook
- `examples/tutorial.ipynb`
  - Clarified that `Context` is used for forward-model examples.
  - Updated the fit examples to use direct `z` / `filters` inputs in:
    - `tf.fit_bol(...)`
    - `tf.fit_multiband(...)`
  - Standardized the tutorial multi-band model key to `nickel`.

## Verification
- Verified with `E:\software\anaconda3\python.exe`.
- Confirmed:
  - `tf.fit_bol(...)` works with direct `z=...`.
  - `tf.fit_multiband(...)` works with direct `z=...` and `filters=...`.
  - Passing `ctx=` into public fit functions now fails at the signature level.
  - Missing `filters` in `fit_multiband(...)` raises a clear error.

## Notes
- Forward-model interfaces and `Context` remain available for advanced usage.
- This change keeps the API explicit while still lowering the standard user learning cost.
