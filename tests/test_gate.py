from ci import quality_gate

BASELINE = {"test": {"precision": 0.308, "recall": 0.754, "f1": 0.438}}
GOOD = {"test": {"precision": 0.308, "recall": 0.754, "f1": 0.438},
        "slices": {"month_to_month": {"recall": 0.899, "n_positive": 129}, "long_contract": {"recall": 0.263, "n_positive": 38}}}


def failed(candidate):
    return [r for r, ok, _, _ in quality_gate.evaluate(candidate, BASELINE) if not ok]


def test_the_baseline_itself_passes():
    assert failed(GOOD) == []


# TODO 4b - add at least THREE more tests here, for example:
#   - recall drops to 0.50                  -> the gate must block
#   - F1 drops by 0.01                      -> the gate must NOT block (that is noise)
#   - a slice with only 4 churners, recall 0 -> the gate must NOT block
def test_low_recall_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["test"]["recall"] = 0.50
    assert "test recall >= floor" in failed(bad)


def test_low_precision_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["test"]["precision"] = 0.15
    assert "test precision >= floor" in failed(bad)


def test_f1_drop_beyond_tolerance_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["test"]["f1"] = 0.438 - 0.12
    assert "test F1 not below baseline - tolerance" in failed(bad)


def test_a_small_f1_wobble_is_allowed():
    ok = copy.deepcopy(GOOD)
    ok["test"]["f1"] = 0.438 - 0.01          # test-set noise, not a real drop
    assert failed(ok) == []


def test_big_group_regression_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["slices"]["month_to_month"]["recall"] = 0.55
    assert "month_to_month slice recall >= floor" in failed(bad)


def test_small_group_collapse_is_blocked():
    bad = copy.deepcopy(GOOD)
    bad["slices"]["long_contract"]["recall"] = 0.05
    assert "long_contract slice recall >= floor" in failed(bad)


def test_a_slice_with_four_churners_is_not_enforced():
    ok = copy.deepcopy(GOOD)
    ok["slices"]["long_contract"] = {"recall": 0.0, "n_positive": 4}
    assert failed(ok) == []
