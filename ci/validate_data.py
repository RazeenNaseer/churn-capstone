"""Data checks that run BEFORE training. Exit code 1 if any check fails.

    python ci/validate_data.py data/churn.csv
"""
import sys

import pandas as pd

COLUMNS = ["tenure_months", "monthly_charges", "contract_type", "support_calls_90d", "late_payments_12m",
           "has_security_addon", "age", "churned"]
MIN_ROWS = 1000
CHURN_RATE_RANGE = (0.08, 0.30)        # TODO 1 - is this range sensible? (the real rate is about 17%)
MAX_DUPLICATE_SHARE = 0.05


def run_checks(df):
    """Return a list of (name, passed, detail)."""
    results = []

    def check(name, passed, detail=""):
        results.append((name, bool(passed), detail))

    check("columns match the contract", list(df.columns) == COLUMNS, f"found {list(df.columns)}")
    if list(df.columns) != COLUMNS:
        return results
    check("enough rows", len(df) >= MIN_ROWS, f"{len(df)} rows (minimum {MIN_ROWS})")
    check("no missing values", df.isna().sum().sum() == 0, f"{int(df.isna().sum().sum())} missing")
    check("tenure_months is 0-120", df["tenure_months"].between(0, 120).all(),
          f"range {df['tenure_months'].min()}-{df['tenure_months'].max()}")
    check("monthly_charges is positive", (df["monthly_charges"] > 0).all(), f"min {df['monthly_charges'].min()}")
    check("contract_type is 0, 1 or 2", df["contract_type"].isin([0, 1, 2]).all(),
          f"values {sorted(df['contract_type'].unique().tolist())}")
    check("support_calls_90d is not negative", (df["support_calls_90d"] >= 0).all(), f"min {df['support_calls_90d'].min()}")
    check("late_payments_12m is not negative", (df["late_payments_12m"] >= 0).all(), f"min {df['late_payments_12m'].min()}")
    check("has_security_addon is 0 or 1", df["has_security_addon"].isin([0, 1]).all())
    check("age is 18-110", df["age"].between(18, 110).all(), f"range {df['age'].min()}-{df['age'].max()}")
    check("label is 0 or 1", df["churned"].isin([0, 1]).all())
    rate = df["churned"].mean()
    check("churn rate in expected range", CHURN_RATE_RANGE[0] <= rate <= CHURN_RATE_RANGE[1],
          f"{rate:.3f} (expected {CHURN_RATE_RANGE[0]}-{CHURN_RATE_RANGE[1]})")
    dup = df.duplicated().mean()
    check("few duplicate rows", dup <= MAX_DUPLICATE_SHARE, f"{dup:.3f} (maximum {MAX_DUPLICATE_SHARE})")

    return results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    df = pd.read_csv(argv[0] if argv else "data/churn.csv")
    results = run_checks(df)
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n{len(failed)} data check(s) failed: the model was NOT trained.")
        return 1
    print(f"\nall {len(results)} data checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
