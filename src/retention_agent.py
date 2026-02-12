"""AI-powered retention email generator using Gemini + RAG."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "config"))

import os
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from settings import CHROMA_DIR, RAG_COLLECTION_NAME, GEMINI_MODEL, GEMINI_TEMPERATURE

load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = """You are a senior customer retention specialist at a telecommunications company.
Your goal is to write a warm, personalized email that makes the customer feel valued
and encourages them to stay with the company.

EMAIL REQUIREMENTS:
- Length: 150-200 words
- Tone: Warm, empathetic, professional
- Must reference their specific history/interactions
- Must acknowledge their loyalty (tenure)
- Must present the discount as a thank-you, not desperation
- Include a clear call-to-action
- Sign as "Your Customer Success Team"

AVOID:
- Generic corporate language
- Begging or desperate tone
- Mentioning "churn prediction" or "AI"
- Overly salesy language"""

HUMAN_TEMPLATE = """Generate a retention email for this customer:

Customer ID: {customer_id}
Churn Risk Level: {risk_level}
Tenure: {tenure} months ({years} years)
Monthly Charges: ${monthly_charges:.2f}
Customer Lifetime Value: ${clv:.2f}

CUSTOMER INTERACTION HISTORY:
{history}

OFFER TO INCLUDE:
- {discount}% discount on next 6 months
- Waived activation fee for any service upgrades

Generate the email now:"""


class RetentionAgent:
    """AI-powered retention email generator using RAG."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Add it to your .env file."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=GEMINI_TEMPERATURE,
            google_api_key=api_key,
        )

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(RAG_COLLECTION_NAME)

    def calculate_discount(self, clv: float, churn_probability: float) -> int:
        """Calculate personalized discount based on CLV and churn risk.

        High CLV + High Risk  -> 30%
        Med  CLV + High Risk  -> 20%
        Low  CLV + High Risk  -> 15%
        Default               -> 10%
        """
        if clv > 2000 and churn_probability > 0.7:
            return 30
        elif clv > 1000 and churn_probability > 0.6:
            return 20
        elif churn_probability > 0.5:
            return 15
        return 10

    def retrieve_customer_history(self, customer_id: str) -> dict:
        """Retrieve customer history from ChromaDB."""
        try:
            results = self.collection.get(ids=[customer_id])
            if results and results["documents"]:
                return {
                    "history": results["documents"][0],
                    "metadata": results["metadatas"][0],
                }
        except Exception:
            pass

        return {"history": "No interaction history available.", "metadata": {}}

    def generate_retention_email(
        self,
        customer_id: str,
        churn_probability: float,
        customer_data: dict | None = None,
    ) -> dict:
        """Generate a personalized retention email.

        Args:
            customer_id: Customer identifier.
            churn_probability: 0-1 churn risk score.
            customer_data: Optional dict with tenure, MonthlyCharges, CLV.

        Returns:
            dict with keys: email, discount, clv, tenure.
        """

        context = self.retrieve_customer_history(customer_id)
        history = context["history"]
        metadata = context["metadata"]

        tenure = (customer_data or {}).get("tenure", metadata.get("tenure", 0))
        monthly_charges = (customer_data or {}).get(
            "MonthlyCharges", metadata.get("MonthlyCharges", 0)
        )
        clv = (customer_data or {}).get("CLV", metadata.get("CLV", 0))

        discount = self.calculate_discount(clv, churn_probability)

        # Risk label
        if churn_probability > 0.8:
            risk_level = "Very High"
        elif churn_probability > 0.6:
            risk_level = "High"
        elif churn_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE),
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "customer_id": customer_id,
            "risk_level": risk_level,
            "tenure": tenure,
            "years": round(tenure / 12, 1),
            "monthly_charges": monthly_charges,
            "clv": clv,
            "history": history,
            "discount": discount,
        })

        return {
            "email": response.content,
            "discount": discount,
            "clv": clv,
            "tenure": tenure,
        }


# --- Quick test ---
if __name__ == "__main__":
    agent = RetentionAgent()

    result = agent.generate_retention_email(
        customer_id="7590-VHVEG",
        churn_probability=0.85,
        customer_data={
            "tenure": 24,
            "MonthlyCharges": 75.50,
            "CLV": 1812,
        },
    )

    print("=" * 60)
    print("GENERATED RETENTION EMAIL")
    print("=" * 60)
    print(f"Discount Offered: {result['discount']}%")
    print(f"CLV: ${result['clv']:.2f}")
    print(f"Tenure: {result['tenure']} months\n")
    print(result["email"])
    print("=" * 60)
