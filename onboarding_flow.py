"""
onboarding_flow.py — User Onboarding Conversation Flow

Features:
- Multi-step onboarding questions
- Profile collection via conversation
- Skip/later options
- Quick setup for known entity types

Author: Finvij Team
Phase: 2 (Days 8-9)
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel
from user_profile import UserProfile, get_profile_manager
import logging

log = logging.getLogger("onboarding")

# ================================================================================
# ONBOARDING STATES
# ================================================================================

class OnboardingState(str, Enum):
    """Onboarding conversation states"""
    NOT_STARTED = "not_started"
    ASK_ENTITY_TYPE = "ask_entity_type"
    ASK_LICENSE = "ask_license"
    ASK_PRODUCTS = "ask_products"
    ASK_DIGITAL_LENDING = "ask_digital_lending"
    ASK_SCALE = "ask_scale"
    ASK_GEOGRAPHY = "ask_geography"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class OnboardingSession(BaseModel):
    """Track onboarding progress"""
    user_id: str
    state: OnboardingState = OnboardingState.NOT_STARTED
    collected_data: Dict[str, Any] = {}
    step_count: int = 0


# ================================================================================
# ONBOARDING QUESTIONS
# ================================================================================

ONBOARDING_QUESTIONS = {
    OnboardingState.ASK_ENTITY_TYPE: {
        "question": """To provide you with the most relevant RBI regulatory guidance, I'd like to understand your organization better.

**What type of financial entity are you?**

Please select:
1. NBFC (Non-Banking Financial Company)
2. Bank (Commercial Bank / Scheduled Bank)
3. Fintech / Digital Platform
4. LSP (Lending Service Provider)
5. Cooperative Bank
6. Small Finance Bank
7. Housing Finance Company
8. Other / Skip

You can also type your answer (e.g., "We are an NBFC")""",
        "options": ["NBFC", "Bank", "Fintech", "LSP", "Cooperative_Bank", "Small_Finance_Bank", "HFC", "Skip"],
        "next_state_map": {
            "NBFC": OnboardingState.ASK_LICENSE,
            "Bank": OnboardingState.ASK_PRODUCTS,
            "Fintech": OnboardingState.ASK_DIGITAL_LENDING,
            "LSP": OnboardingState.ASK_DIGITAL_LENDING,
            "Cooperative_Bank": OnboardingState.ASK_PRODUCTS,
            "Small_Finance_Bank": OnboardingState.ASK_PRODUCTS,
            "HFC": OnboardingState.ASK_PRODUCTS,
            "Skip": OnboardingState.SKIPPED,
        }
    },

    OnboardingState.ASK_LICENSE: {
        "question": """**What is your NBFC license category?**

Please select:
1. NBFC-ICC (Investment and Credit Company)
2. NBFC-MFI (Microfinance Institution)
3. NBFC-P2P (Peer to Peer Lending)
4. NBFC-Factor
5. NBFC-IDF (Infrastructure Debt Fund)
6. Other / Not sure

You can also type: "We are an NBFC-ICC" """,
        "options": ["NBFC-ICC", "NBFC-MFI", "NBFC-P2P", "NBFC-Factor", "NBFC-IDF", "Other", "Skip"],
        "next_state": OnboardingState.ASK_PRODUCTS
    },

    OnboardingState.ASK_PRODUCTS: {
        "question": """**What financial products do you offer?** (Select all that apply)

You can choose multiple:
1. Gold Loans
2. Personal Loans
3. Microfinance / JLG Loans
4. Housing / Home Loans
5. Vehicle / Auto Loans
6. Business / MSME Loans
7. Other
8. Skip

Type numbers separated by commas (e.g., "1,2,3") or describe: "We offer gold loans and personal loans" """,
        "options": ["gold_loans", "personal_loans", "microfinance", "housing_loans", "vehicle_loans", "business_loans", "other", "Skip"],
        "next_state": OnboardingState.ASK_DIGITAL_LENDING,
        "multi_select": True
    },

    OnboardingState.ASK_DIGITAL_LENDING: {
        "question": """**Do you engage in digital lending?**

This includes:
- App-based lending
- Online loan applications
- Use of LSPs (Lending Service Providers)
- FLDG/DLG arrangements

Please answer:
1. Yes, we do digital lending
2. Yes, with FLDG/DLG arrangements
3. Yes, through LSP
4. No
5. Skip

You can also type: "Yes, we use FLDG" """,
        "options": ["yes", "yes_with_fldg", "yes_via_lsp", "no", "Skip"],
        "next_state": OnboardingState.ASK_SCALE
    },

    OnboardingState.ASK_SCALE: {
        "question": """**What is your organization's scale?**

Please select:
1. Small (< ₹100 Cr assets)
2. Medium (₹100 Cr - ₹500 Cr)
3. Large (₹500 Cr - ₹5,000 Cr)
4. Very Large (> ₹5,000 Cr)
5. Skip

You can also type: "We are a medium-sized NBFC" """,
        "options": ["small", "medium", "large", "very_large", "Skip"],
        "next_state": OnboardingState.ASK_GEOGRAPHY
    },

    OnboardingState.ASK_GEOGRAPHY: {
        "question": """**What geographies do you operate in?** (Select all that apply)

1. Urban areas
2. Rural areas
3. Semi-urban
4. Pan-India
5. Skip

Type numbers separated by commas or describe: "We operate in urban and rural areas" """,
        "options": ["urban", "rural", "semi_urban", "pan_india", "Skip"],
        "next_state": OnboardingState.COMPLETED,
        "multi_select": True
    },
}


# ================================================================================
# ONBOARDING HANDLER
# ================================================================================

class OnboardingHandler:
    """
    Manages onboarding conversation flow.
    """

    def __init__(self):
        self.sessions: Dict[str, OnboardingSession] = {}

    def start_onboarding(self, user_id: str) -> str:
        """
        Start onboarding for a user.

        Returns:
            First onboarding question
        """
        session = OnboardingSession(
            user_id=user_id,
            state=OnboardingState.ASK_ENTITY_TYPE
        )
        self.sessions[user_id] = session

        question_data = ONBOARDING_QUESTIONS[OnboardingState.ASK_ENTITY_TYPE]
        return question_data["question"]

    def is_onboarding_active(self, user_id: str) -> bool:
        """Check if user is in onboarding flow"""
        session = self.sessions.get(user_id)
        if not session:
            return False

        return session.state not in (
            OnboardingState.NOT_STARTED,
            OnboardingState.COMPLETED,
            OnboardingState.SKIPPED
        )

    def process_response(self, user_id: str, response: str) -> Dict[str, Any]:
        """
        Process user response to onboarding question.

        Returns:
            Dict with:
            - next_question: Next question to ask (or None if done)
            - completed: Whether onboarding is complete
            - profile_updates: Profile data collected
        """
        session = self.sessions.get(user_id)
        if not session:
            return {
                "next_question": None,
                "completed": False,
                "error": "No active onboarding session"
            }

        current_state = session.state
        question_data = ONBOARDING_QUESTIONS.get(current_state)

        if not question_data:
            return {
                "next_question": None,
                "completed": True,
                "profile_updates": session.collected_data
            }

        # Parse response
        parsed = self._parse_response(response, question_data)

        # Handle skip
        if parsed == "Skip" or "skip" in response.lower():
            session.state = OnboardingState.SKIPPED
            return {
                "next_question": None,
                "completed": True,
                "skipped": True,
                "message": "Onboarding skipped. You can always update your profile later using the /profile command.",
                "profile_updates": session.collected_data
            }

        # Collect data
        field_name = self._get_field_name(current_state)
        session.collected_data[field_name] = parsed

        # Determine next state
        next_state = self._get_next_state(current_state, parsed, question_data)
        session.state = next_state
        session.step_count += 1

        # Check if completed
        if next_state == OnboardingState.COMPLETED:
            return {
                "next_question": None,
                "completed": True,
                "message": "✓ Profile setup complete! I'll now provide personalized regulatory guidance based on your profile.",
                "profile_updates": session.collected_data
            }

        # Get next question
        next_question_data = ONBOARDING_QUESTIONS.get(next_state)
        if next_question_data:
            return {
                "next_question": next_question_data["question"],
                "completed": False,
                "profile_updates": session.collected_data
            }

        # Fallback: complete
        session.state = OnboardingState.COMPLETED
        return {
            "next_question": None,
            "completed": True,
            "profile_updates": session.collected_data
        }

    def _parse_response(self, response: str, question_data: Dict) -> Any:
        """Parse user response based on question type"""
        response_lower = response.lower().strip()

        # Check for skip
        if "skip" in response_lower or "later" in response_lower:
            return "Skip"

        # Check against options
        options = question_data.get("options", [])
        is_multi = question_data.get("multi_select", False)

        # Try exact option match
        for option in options:
            if option.lower() in response_lower or response_lower in option.lower():
                if is_multi:
                    return [option]
                return option

        # Try number-based selection
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(options):
                option = options[idx]
                if is_multi:
                    return [option]
                return option

        # For multi-select, try comma-separated numbers
        if is_multi and ',' in response:
            selected = []
            for part in response.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(options):
                        selected.append(options[idx])
            if selected:
                return selected

        # Fallback: return raw response for inference
        return response

    def _get_field_name(self, state: OnboardingState) -> str:
        """Map state to profile field name"""
        mapping = {
            OnboardingState.ASK_ENTITY_TYPE: "entity_type",
            OnboardingState.ASK_LICENSE: "license_category",
            OnboardingState.ASK_PRODUCTS: "products",
            OnboardingState.ASK_DIGITAL_LENDING: "digital_lending",
            OnboardingState.ASK_SCALE: "asset_size",
            OnboardingState.ASK_GEOGRAPHY: "geography",
        }
        return mapping.get(state, "unknown")

    def _get_next_state(
        self,
        current_state: OnboardingState,
        response: Any,
        question_data: Dict
    ) -> OnboardingState:
        """Determine next onboarding state"""

        # Check for state map (conditional next state)
        next_state_map = question_data.get("next_state_map")
        if next_state_map and isinstance(response, str):
            return next_state_map.get(response, question_data.get("next_state", OnboardingState.COMPLETED))

        # Default next state
        return question_data.get("next_state", OnboardingState.COMPLETED)

    def complete_onboarding(self, user_id: str) -> UserProfile:
        """
        Complete onboarding and create/update user profile.
        """
        session = self.sessions.get(user_id)
        if not session:
            raise ValueError(f"No onboarding session for user {user_id}")

        # Get profile manager
        pm = get_profile_manager()

        # Parse collected data into profile fields
        profile_data = self._parse_collected_data(session.collected_data)

        # Create or update profile
        profile = pm.get_or_create(user_id, **profile_data)

        # Update with collected data
        for key, value in profile_data.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        # Save profile
        pm.store.save(profile)

        # Mark session as completed
        session.state = OnboardingState.COMPLETED

        log.info(f"Onboarding completed for user {user_id}")
        return profile

    def _parse_collected_data(self, collected: Dict[str, Any]) -> Dict[str, Any]:
        """Parse collected data into profile-compatible format"""
        parsed = {}

        # Entity type
        if "entity_type" in collected:
            parsed["entity_type"] = collected["entity_type"]

        # License category
        if "license_category" in collected:
            parsed["license_category"] = collected["license_category"]

        # Products
        if "products" in collected:
            products = collected["products"]
            if isinstance(products, list):
                parsed["products"] = products
            else:
                parsed["products"] = [products]

        # Digital lending
        if "digital_lending" in collected:
            dl_response = collected["digital_lending"]
            if isinstance(dl_response, str):
                dl_response_lower = dl_response.lower()
                parsed["digital_lending"] = "yes" in dl_response_lower

                if "fldg" in dl_response_lower or "dlg" in dl_response_lower:
                    parsed["has_fldg_arrangements"] = True

                if "lsp" in dl_response_lower:
                    parsed["uses_lsp"] = True

        # Scale
        if "asset_size" in collected:
            parsed["asset_size"] = collected["asset_size"]

        # Geography
        if "geography" in collected:
            geo = collected["geography"]
            if isinstance(geo, list):
                parsed["geography"] = geo
            else:
                parsed["geography"] = [geo]

        return parsed


# ================================================================================
# GLOBAL INSTANCE
# ================================================================================

_onboarding_handler = None

def get_onboarding_handler() -> OnboardingHandler:
    """Get global onboarding handler instance"""
    global _onboarding_handler

    if _onboarding_handler is None:
        _onboarding_handler = OnboardingHandler()

    return _onboarding_handler
