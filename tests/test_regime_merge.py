"""Regime detection: when HMM and volatility disagree, the riskier label wins."""
import numpy as np
import pandas as pd
import pytest

from core.regime_detector import CRISIS, HIGH_VOLATILITY, RANGING, TRENDING, RegimeDetector

ORDER = [TRENDING, RANGING, HIGH_VOLATILITY, CRISIS]


@pytest.mark.parametrize("a", ORDER)
@pytest.mark.parametrize("b", ORDER)
def test_conservative_merge_picks_riskier_label(a, b):
    expected = ORDER[max(ORDER.index(a), ORDER.index(b))]
    assert RegimeDetector()._conservative_merge(a, b) == expected


def test_detect_uses_merge_on_disagreement(monkeypatch):
    d = RegimeDetector()
    monkeypatch.setattr(d, "_extract_features", lambda df: np.zeros((1, 1)))
    monkeypatch.setattr(d, "_hmm_classify", lambda f: TRENDING)
    monkeypatch.setattr(d, "_volatility_regime", lambda df: HIGH_VOLATILITY)
    prices = pd.DataFrame({"close": np.linspace(100, 110, 40)})

    assert d.detect(prices) == HIGH_VOLATILITY
    assert d.regime_history[-1]["hmm_regime"] == TRENDING
    assert d.regime_history[-1]["vol_regime"] == HIGH_VOLATILITY
