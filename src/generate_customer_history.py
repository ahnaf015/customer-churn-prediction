"""Generate simulated customer interaction histories for the RAG system."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import pandas as pd
import random
from datetime import datetime, timedelta

from settings import RAW_DATA_FILE, CUSTOMER_HISTORIES_FILE, HISTORY_SAMPLE_SIZE

# Fix seed for reproducibility
random.seed(42)

# --- Interaction templates ---

SUPPORT_TEMPLATES = [
    "Called support about billing discrepancy on {date}. Issue resolved.",
    "Complained about slow internet speeds on {date}. Technician scheduled.",
    "Requested service upgrade quote on {date}.",
    "Asked about contract early termination fees on {date}.",
    "Reported outage on {date}. Service restored within 2 hours.",
    "Inquired about adding family members to plan on {date}.",
    "Requested device protection activation on {date}.",
    "Called about unexpected price increase on {date}. Credited $15.",
    "Reported intermittent connection drops on {date}. Router replaced.",
]

PURCHASE_TEMPLATES = [
    "Upgraded to premium internet plan ({speed}Mbps) on {date}.",
    "Added phone service package on {date}.",
    "Purchased device protection ($10/month) on {date}.",
    "Renewed contract for {contract_length} on {date}.",
    "Subscribed to streaming service bundle on {date}.",
    "Added online security suite on {date}.",
    "Upgraded to paperless billing on {date}.",
]

USAGE_TEMPLATES = [
    "Heavy internet usage in {month} (avg {usage}GB/day).",
    "Minimal service usage in last 3 months (avg {usage}GB/day).",
    "Consistent monthly data usage around {usage}GB.",
    "Usage increased {percent}% compared to previous quarter.",
    "Frequent streaming activity detected in {month}.",
    "Multiple device connections observed ({devices} devices).",
]

FEEDBACK_TEMPLATES = [
    "Left positive review mentioning customer service on {date}.",
    "Submitted complaint about unexpected charges on {date}.",
    "Participated in customer satisfaction survey - rated {rating}/5 on {date}.",
    "Referred {referrals} friends to service in {month}.",
    "Contacted via social media about service issues on {date}.",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _random_date() -> str:
    days_ago = random.randint(0, 730)
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _random_month() -> str:
    return random.choice(MONTHS)


def _build_interactions() -> str:
    """Build a single customer's interaction history string."""

    interactions = []

    # Support interactions (2-3)
    for _ in range(random.randint(2, 3)):
        interactions.append(
            random.choice(SUPPORT_TEMPLATES).format(date=_random_date())
        )

    # Purchase history (1-2)
    for _ in range(random.randint(1, 2)):
        interactions.append(
            random.choice(PURCHASE_TEMPLATES).format(
                date=_random_date(),
                speed=random.choice([50, 100, 200, 500]),
                contract_length=random.choice(["1 year", "2 years"]),
            )
        )

    # Usage patterns (1-2)
    for _ in range(random.randint(1, 2)):
        interactions.append(
            random.choice(USAGE_TEMPLATES).format(
                month=_random_month(),
                usage=random.randint(20, 150),
                percent=random.randint(10, 80),
                devices=random.randint(2, 8),
            )
        )

    # Feedback (50% chance)
    if random.random() > 0.5:
        interactions.append(
            random.choice(FEEDBACK_TEMPLATES).format(
                date=_random_date(),
                month=_random_month(),
                rating=random.randint(1, 5),
                referrals=random.randint(0, 3),
            )
        )

    return " | ".join(interactions)


def generate_customer_histories(df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    """Generate simulated histories for the first n_samples customers."""

    records = []
    for _, row in df.head(n_samples).iterrows():
        records.append({
            "customerID": row["customerID"],
            "history": _build_interactions(),
            "tenure": row["tenure"],
            "MonthlyCharges": row["MonthlyCharges"],
            "TotalCharges": row["TotalCharges"],
            "CLV": row["tenure"] * row["MonthlyCharges"],
            "Contract": row["Contract"],
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Loading customer data...")
    df = pd.read_csv(RAW_DATA_FILE)
    # Fix TotalCharges for CLV calculation
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    print(f"Generating {HISTORY_SAMPLE_SIZE} customer histories...")
    histories = generate_customer_histories(df, n_samples=HISTORY_SAMPLE_SIZE)

    histories.to_csv(CUSTOMER_HISTORIES_FILE, index=False)
    print(f"Saved {len(histories)} histories to {CUSTOMER_HISTORIES_FILE}")

    print(f"\nSample history:\n{histories.iloc[0]['history']}")
