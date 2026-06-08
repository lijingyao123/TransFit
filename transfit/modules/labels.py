from __future__ import annotations


def normalize_band_label(label: object) -> str:
    """
    Normalize a photometric band label.

    Band matching is intentionally case-sensitive; only surrounding whitespace
    is stripped.
    """
    out = str(label).strip()
    if not out:
        raise ValueError("Band labels must be non-empty strings.")
    return out


__all__ = ["normalize_band_label"]
