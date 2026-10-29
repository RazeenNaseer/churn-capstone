import numpy as np
import pandas as pd

from ci import validate_data


def good_frame(n=1500):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n), "monthly_charges": rng.uniform(20, 120, n) + rng.random(n) * 1e-3,
        "contract_type": rng.integers(0, 3, n), "support_calls_90d": rng.poisson(1, n), "late_payments_12m": rng.poisson(0.5, n),
        "has_security_addon": rng.integers(0, 2, n), "age": rng.integers(18, 80, n),
        "churned": (rng.random(n) < 0.17).astype(int),
    })


def failed(df):
    return {name for name, ok, _ in validate_data.run_checks(df) if not ok}


def test_a_clean_frame_passes():
    assert failed(good_frame()) == set()


# TODO 4a - add at least THREE more tests here. One test per bad-data case, for example:
#   def test_negative_tenure_is_caught():
#       assert failed(good_frame().assign(tenure_months=-3)) != set()
def test_negative_tenure_is_caught():
    assert "tenure_months is 0-120" in failed(good_frame().assign(tenure_months=-3))


def test_tenure_above_120_is_caught():
    assert "tenure_months is 0-120" in failed(good_frame().assign(tenure_months=500))


def test_unknown_contract_type_is_caught():
    assert "contract_type is 0, 1 or 2" in failed(good_frame().assign(contract_type=5))


def test_zero_monthly_charge_is_caught():
    assert "monthly_charges is positive" in failed(good_frame().assign(monthly_charges=0.0))


def test_negative_support_calls_are_caught():
    assert "support_calls_90d is not negative" in failed(good_frame().assign(support_calls_90d=-1))


def test_age_below_18_is_caught():
    assert "age is 18-110" in failed(good_frame().assign(age=7))


def test_label_value_2_is_caught():
    assert "label is 0 or 1" in failed(good_frame().assign(churned=2))


def test_a_file_with_no_churners_is_caught():
    assert "churn rate in expected range" in failed(good_frame().assign(churned=0))


def test_missing_values_are_caught():
    df = good_frame()
    df.loc[0, "monthly_charges"] = np.nan
    assert "no missing values" in failed(df)


def test_heavy_duplication_is_caught():
    df = good_frame()
    assert "few duplicate rows" in failed(pd.concat([df, df.head(500)]))


def test_too_few_rows_are_caught():
    assert "enough rows" in failed(good_frame().head(100))


def test_a_missing_column_is_caught():
    assert "columns match the contract" in failed(good_frame().drop(columns=["age"]))
