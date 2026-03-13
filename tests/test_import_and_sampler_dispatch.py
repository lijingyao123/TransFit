import importlib
import sys

import pytest
import transfit
from transfit.api import _run_sampler


class _DummyPrior:
    param_names = ["x"]


def test_import_transfit_exposes_public_api():
    assert hasattr(transfit, "fit_bol")
    assert hasattr(transfit, "fit_multiband")
    assert hasattr(transfit, "BolometricData")
    assert hasattr(transfit, "MultiBandData")


def test_sampler_backends_are_not_imported_on_samplers_import():
    sys.modules.pop("transfit.samplers.emcee", None)
    sys.modules.pop("transfit.samplers.zeus", None)
    sys.modules.pop("transfit.samplers.dynesty", None)

    importlib.import_module("transfit.samplers")

    assert "transfit.samplers.emcee" not in sys.modules
    assert "transfit.samplers.zeus" not in sys.modules
    assert "transfit.samplers.dynesty" not in sys.modules


def test_run_sampler_rejects_unknown_sampler():
    with pytest.raises(ValueError):
        _run_sampler(
            sampler="unknown",
            lnprob=lambda _: 0.0,
            prior=_DummyPrior(),
            sampler_kwargs={},
        )
