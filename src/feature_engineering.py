import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced features for churn prediction.

    Expects data AFTER clean_data() but BEFORE encode_categoricals(),
    so categorical columns still contain original string values.
    """

    df = df.copy()

    # 1. Tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )

    # 2. Monthly charges bins
    df["charges_bin"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 100],
        labels=["Low", "Medium", "High"],
    )

    # 3. Customer Lifetime Value
    df["CLV"] = df["tenure"] * df["MonthlyCharges"]

    # 4. Average monthly spend (normalized)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 5. Total number of subscribed services
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[service_cols].apply(
        lambda row: sum(val not in ("No", "No internet service", "No phone service") for val in row),
        axis=1,
    )

    # 6. Contract commitment score
    contract_map = {"Month-to-month": 1, "One year": 2, "Two year": 3}
    df["ContractValue"] = df["Contract"].map(contract_map)

    # 7. Payment method risk (electronic check correlates with higher churn)
    df["PaymentRisk"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    # 8. Senior citizen with high spend
    df["SeniorHighSpend"] = (
        (df["SeniorCitizen"] == 1) & (df["MonthlyCharges"] > 70)
    ).astype(int)

    # 9. No tech support indicator
    df["NoTechSupport"] = df["TechSupport"].isin(["No", "No internet service"]).astype(int)

    # 10. No online security indicator
    df["NoOnlineSecurity"] = df["OnlineSecurity"].isin(["No", "No internet service"]).astype(int)

    # 11. Charge per service (cost efficiency)
    df["ChargePerService"] = df["MonthlyCharges"] / (df["ServiceCount"] + 1)

    # 12. Tenure-charge interaction (long tenure + high charge = loyal high-value)
    df["TenureChargeInteraction"] = df["tenure"] * df["MonthlyCharges"] / 100

    # 13. New customer flag (tenure <= 6 months)
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    # 14. Has partner or dependents (family anchor)
    if df["Partner"].dtype == object:
        df["HasFamily"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    else:
        df["HasFamily"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)

    # 15. Month-to-month + electronic check (high risk combo)
    df["HighRiskCombo"] = (
        (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # 16. Streaming bundle user
    if df["StreamingTV"].dtype == object:
        df["StreamingBundle"] = (
            (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
        ).astype(int)
    else:
        df["StreamingBundle"] = ((df["StreamingTV"] == 1) & (df["StreamingMovies"] == 1)).astype(int)

    # 17. No protection services
    df["NoProtection"] = (
        df["OnlineSecurity"].isin(["No", "No internet service"]) &
        df["DeviceProtection"].isin(["No", "No internet service"]) &
        df["OnlineBackup"].isin(["No", "No internet service"])
    ).astype(int)

    return df
