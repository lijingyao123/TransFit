import numpy as np
import pytest

from transfit.data import BolometricData, MultiBandData


def test_bolometric_data_length_validation():
    with pytest.raises(ValueError):
        BolometricData(t_days=[0.0, 1.0], y=[1.0], yerr=[0.1, 0.2])


def test_bolometric_data_filtered_keeps_only_finite_positive_error():
    data = BolometricData(
        t_days=np.array([0.0, 1.0, 2.0]),
        y=np.array([1.0, np.nan, 3.0]),
        yerr=np.array([0.1, 0.2, -0.5]),
    )
    filtered = data.filtered()

    assert filtered.t_days.tolist() == [0.0]
    assert filtered.y.tolist() == [1.0]
    assert filtered.yerr.tolist() == [0.1]


def test_multiband_data_filtered_applies_mask_and_quality_checks():
    data = MultiBandData(
        t_days=np.array([0.0, 1.0, 2.0]),
        band=np.array(["g", "r", "g"], dtype=object),
        y=np.array([20.1, np.nan, 20.3]),
        yerr=np.array([0.1, 0.2, 0.3]),
        mask=np.array([True, True, False]),
    )
    filtered = data.filtered()

    assert filtered.t_days.tolist() == [0.0]
    assert filtered.band.tolist() == ["g"]
    assert filtered.y.tolist() == [20.1]
    assert filtered.yerr.tolist() == [0.1]


def test_multiband_bands_property_sorted_unique():
    data = MultiBandData(
        t_days=np.array([0.0, 1.0, 2.0]),
        band=np.array(["r", "g", "r"], dtype=object),
        y=np.array([20.0, 20.1, 20.2]),
        yerr=np.array([0.1, 0.1, 0.1]),
    )

    assert data.bands == ["g", "r"]
