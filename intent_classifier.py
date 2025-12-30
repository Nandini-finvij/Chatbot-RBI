# intent_classifier.py
# ---------------------------------------------------------
# LLM-native Intent + Topic Classifier (KG-anchored)
# ---------------------------------------------------------
# ✔ No keyword rules
# ✔ No regex
# ✔ One LLM call
# ✔ Strict JSON
# ✔ Topic constrained to available RBI documents
# ✔ Defensive parsing (markdown / null-safe)
# ---------------------------------------------------------

import json
from groq import Groq


LLM_MODEL = "llama-3.3-70b-versatile"


INTENT_PROMPT = """
You are an RBI regulatory and financial assistant classifier.

Analyze the user query and return a JSON object ONLY.

Your task:
1. Identify the user's primary goal
2. Identify the most relevant RBI topic (if applicable)
3. Identify what user context is required (if any)
4. Provide a confidence score

--------------------------------------------------
Allowed user_goal values:
- chit_chat
- regulatory_query
- financial_planning
- compliance_plan_generation  (NEW: when user asks to "generate compliance plan", "create my compliance plan", "what's my compliance roadmap", etc.)
- clarification
- comparison

--------------------------------------------------
Allowed topic_key values
(choose ONLY from this list, or null if not applicable):

Gold_Loan_LTV
Gold_Loan_Operational

DLG_Cap
DLG_Eligibility
DLG_Forms
DLG_Structure
DLG_Restrictions
DLG_Reporting

Digital_Lending_Core
FAQs_Digital_Lending

ECL_Overview
ECL_Stages
ECL_Measurement

KYC_Process
FAQs_KYC
AML_Compliance

IRACP_Classification
IRACP_Provisioning

Model_Governance_Framework
Model_Governance_Validation

--------------------------------------------------
Allowed context_required values:
- income
- expenses
- assets
- liabilities
- risk_profile
- time_horizon

--------------------------------------------------
Rules:
- Return VALID JSON ONLY
- No markdown, no backticks
- No explanations
- topic_key MUST be null for chit_chat
- context_required MUST be an array (empty if none)

--------------------------------------------------
User query:
\"\"\"{query}\"\"\"
"""


class IntentClassifier:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def classify(self, query: str) -> dict:
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": INTENT_PROMPT.format(query=query)}
            ],
            temperature=0,
            max_tokens=180
        )

        raw = response.choices[0].message.content.strip()

        # -------------------------------------------------
        # Defensive cleanup (LLMs may still do this)
        # -------------------------------------------------
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM:\n{raw}") from e

        # -------------------------------------------------
        # Hard normalization (production safety)
        # -------------------------------------------------
        if data.get("context_required") is None:
            data["context_required"] = []

        if not isinstance(data.get("context_required"), list):
            data["context_required"] = []

        data.setdefault("topic_key", None)
        data.setdefault("confidence", 0.5)

        if "user_goal" not in data:
            raise ValueError(f"user_goal missing in LLM output:\n{data}")

        return data
