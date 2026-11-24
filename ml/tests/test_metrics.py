import numpy as np
import pytest
from scipy.special import expit
from sklearn import metrics as skm

from ml.evaluation import metrics as m


def test_hand_worked_confusion_matrix_case():
    # y:    1    1    1    0    0    0    0    1
    # p:   .9   .8   .3   .6   .2   .1   .4   .7   threshold .5 -> pred 1 1 0 1 0 0 0 1
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4, 0.7])
    r = m.classification_metrics(y, p)
    assert r["confusion_matrix"] == [[3, 1], [1, 3]]  # tn, fp / fn, tp
    assert r["accuracy"] == 6 / 8 and r["precision"] == 3 / 4 and r["recall"] == 3 / 4
    assert r["f1"] == pytest.approx(0.75) and r["specificity"] == 3 / 4
    assert r["false_positive_rate"] == 1 / 4 and r["false_negative_rate"] == 1 / 4
    assert r["n"] == 8


def test_metrics_match_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.25, 500), 0.001, 0.999)
    r = m.classification_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    assert r["accuracy"] == pytest.approx(skm.accuracy_score(y, pred))
    assert r["precision"] == pytest.approx(skm.precision_score(y, pred))
    assert r["recall"] == pytest.approx(skm.recall_score(y, pred))
    assert r["f1"] == pytest.approx(skm.f1_score(y, pred))
    assert r["roc_auc"] == pytest.approx(skm.roc_auc_score(y, p))
    assert r["pr_auc"] == pytest.approx(skm.average_precision_score(y, p))
    assert r["log_loss"] == pytest.approx(skm.log_loss(y, p), rel=1e-4)
    assert r["brier"] == pytest.approx(skm.brier_score_loss(y, p))
    tn, fp, fn, tp = skm.confusion_matrix(y, pred).ravel()
    assert r["confusion_matrix"] == [[tn, fp], [fn, tp]]


def test_degenerate_inputs_do_not_crash_or_divide_by_zero():
    r = m.classification_metrics(np.zeros(5), np.full(5, 0.1))
    assert r["roc_auc"] is None and r["pr_auc"] is None and r["recall"] == 0.0 and r["precision"] == 0.0
    assert r["specificity"] == 1.0 and r["false_positive_rate"] == 0.0
    r = m.classification_metrics(np.ones(4), np.ones(4))
    assert r["recall"] == 1.0 and r["false_negative_rate"] == 0.0 and np.isfinite(r["log_loss"])


def test_threshold_moves_the_decision_boundary():
    y, p = np.array([0, 1]), np.array([0.4, 0.6])
    assert m.classification_metrics(y, p, 0.5)["accuracy"] == 1.0
    assert m.classification_metrics(y, p, 0.7)["accuracy"] == 0.5


def test_ece_is_zero_for_perfect_calibration_and_large_for_overconfidence():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.5, 1.0, 20000)
    y = (rng.random(20000) < p).astype(int)  # outcomes drawn from the stated probability
    assert m.expected_calibration_error(p, y) < 0.02
    overconfident = np.where(y == 1, 0.99, 0.01) * 0 + 0.99  # always claims 99% AI
    labels = (rng.random(20000) < 0.6).astype(int)
    assert m.expected_calibration_error(overconfident, labels) == pytest.approx(0.99 - 0.6, abs=0.02)


def test_ece_hand_case():
    # one bin: confidence .8 on 4 samples, 3 correct -> |0.75 - 0.8| = 0.05
    p = np.array([0.8, 0.8, 0.8, 0.8])
    y = np.array([1, 1, 1, 0])
    assert m.expected_calibration_error(p, y) == pytest.approx(0.05)


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(2)
    true_logits = rng.normal(0, 2, 50000)
    y = (rng.random(50000) < expit(true_logits)).astype(int)
    for temperature in (0.5, 2.0, 3.0):
        t = m.fit_temperature(true_logits * temperature, y)  # model over/under-confident by `temperature`
        assert t == pytest.approx(temperature, rel=0.08)


def test_temperature_scaling_improves_log_loss():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < expit(z)).astype(int)
    over = z * 3
    t = m.fit_temperature(over, y)
    before = skm.log_loss(y, expit(over))
    after = skm.log_loss(y, expit(over / t))
    assert after < before and t > 1


def test_choose_band_respects_error_budget_on_calibration_data():
    rng = np.random.default_rng(4)
    real = np.clip(rng.normal(0.25, 0.2, 4000), 0, 1)
    ai = np.clip(rng.normal(0.75, 0.2, 4000), 0, 1)
    p, y = np.concatenate([real, ai]), np.r_[np.zeros(4000), np.ones(4000)]
    low, high = m.choose_band(p, y, max_error=0.02)
    assert low <= 0.5 <= high
    assert (real >= high).mean() <= 0.02 + 1e-9 and (ai <= low).mean() <= 0.02 + 1e-9
    stats = m.band_metrics(p, y, low, high)
    assert stats["false_positive_rate_on_real"] <= 0.02 + 1e-9 and 0 < stats["coverage"] < 1


def test_band_metrics_hand_case():
    p = np.array([0.05, 0.3, 0.5, 0.7, 0.95, 0.9])
    y = np.array([0, 0, 1, 0, 1, 0])
    s = m.band_metrics(p, y, low=0.2, high=0.8)
    # decided: .05(real ok), .95(ai ok), .9(AI predicted, truth real -> error); inconclusive: .3,.5,.7
    assert s["inconclusive"] == 3 and s["coverage"] == 0.5
    assert s["accuracy_on_decided"] == pytest.approx(2 / 3)
    assert s["false_positive_rate_on_real"] == 1 / 4 and s["error_rate"] == 1 / 6


def test_wide_band_when_distributions_overlap_and_perfect_when_separated():
    y = np.array([0] * 50 + [1] * 50)
    separated = np.r_[np.full(50, 0.1), np.full(50, 0.9)]
    low, high = m.choose_band(separated, y, 0.0)
    assert m.band_metrics(separated, y, low, high)["coverage"] == 1.0
    overlapped = np.r_[np.linspace(0.3, 0.7, 50), np.linspace(0.3, 0.7, 50)]
    low, high = m.choose_band(overlapped, y, 0.0)
    assert m.band_metrics(overlapped, y, low, high)["coverage"] < 0.2


def test_asymmetric_confusion_counts_pin_every_rate_formula():
    # negatives: 4 correct (.1), 1 false positive (.9); positives: 2 correct (.9), 2 missed (.4)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.4, 0.4])
    r = m.classification_metrics(y, p)
    assert r["confusion_matrix"] == [[4, 1], [2, 2]]
    assert r["specificity"] == 4 / 5 and r["false_positive_rate"] == 1 / 5
    assert r["false_negative_rate"] == 2 / 4 and r["recall"] == 2 / 4
    assert r["precision"] == 2 / 3 and r["accuracy"] == 6 / 9


def test_probability_exactly_at_threshold_counts_as_positive():
    r = m.classification_metrics(np.array([1, 0]), np.array([0.5, 0.5]))
    assert r["confusion_matrix"] == [[0, 1], [0, 1]]


def test_ece_is_symmetric_in_class_labels():
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 3000)
    y = (rng.random(3000) < 0.5 * p + 0.25).astype(int)
    assert m.expected_calibration_error(p, y) == pytest.approx(m.expected_calibration_error(1 - p, 1 - y))
    assert m.expected_calibration_error(p, y) > 0.02  # confidently-low probabilities are judged too


def test_ece_uses_fine_bins_between_half_and_one():
    # confidences .68 (correct) and .71 (wrong) fall in different 15-bin cells over [.5, 1] -> .5*.32 + .5*.71
    p = np.array([0.68, 0.71])
    y = np.array([1, 0])
    assert m.expected_calibration_error(p, y) == pytest.approx(0.5 * 0.32 + 0.5 * 0.71)


def test_band_edges_are_inclusive_and_clamped_around_half():
    s = m.band_metrics(np.array([0.2, 0.8]), np.array([0, 1]), low=0.2, high=0.8)
    assert s["coverage"] == 1.0 and s["accuracy_on_decided"] == 1.0
    y = np.array([0] * 20 + [1] * 20)
    separated = np.r_[np.full(20, 0.1), np.full(20, 0.9)]
    assert m.choose_band(separated, y, 0.0) == (0.5, 0.5)  # crossing cut-offs collapse to the neutral midpoint
    s = m.band_metrics(separated, y, 0.5, 0.5)
    assert s["accuracy_on_decided"] == 1.0 and s["error_rate"] == 0.0
