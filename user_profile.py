"""
user_profile.py — User Profiling System for RBI Chatbot

Features:
- User profile schema with validation
- Profile storage (Redis + JSON fallback)
- Profile inference from queries
- Profile-aware retrieval & re-ranking
- Onboarding conversation flow

Author: Finvij Team
Phase: 2 (Days 6-12)
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json
import logging
from pathlib import Path

log = logging.getLogger("user_profile")

# ================================================================================
# USER PROFILE SCHEMA
# ================================================================================

class UserProfile(BaseModel):
    """
    User profile for personalized RBI regulatory assistance.

    Captures entity details, business context, and compliance needs
    to provide tailored regulatory guidance.
    """

    # Core Identity
    user_id: str = Field(..., description="Unique user identifier")
    entity_name: Optional[str] = Field(None, description="Organization name")

    # Entity Classification
    entity_type: str = Field(
        ...,
        description="Type of financial entity"
    )
    license_category: Optional[str] = Field(
        None,
        description="Specific license/registration category"
    )

    # Business Scale
    asset_size: Optional[str] = Field(
        None,
        description="Asset size category"
    )

    # Products & Services
    products: List[str] = Field(
        default_factory=list,
        description="Financial products offered"
    )

    # Digital Lending
    digital_lending: bool = Field(
        default=False,
        description="Whether entity does digital lending"
    )
    has_fldg_arrangements: bool = Field(
        default=False,
        description="Whether entity has FLDG/DLG arrangements"
    )
    uses_lsp: bool = Field(
        default=False,
        description="Whether entity uses Lending Service Providers"
    )

    # Geography
    geography: List[str] = Field(
        default_factory=list,
        description="Operating regions"
    )

    # Compliance Areas (auto-detected)
    compliance_areas: List[str] = Field(
        default_factory=list,
        description="Detected compliance focus areas"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    query_count: int = Field(default=0, description="Number of queries asked")

    @validator('entity_type')
    def validate_entity_type(cls, v):
        """Validate entity type against allowed values"""
        allowed = {
            "NBFC", "Bank", "Fintech", "LSP", "Cooperative_Bank",
            "Small_Finance_Bank", "Payment_Bank", "ARC", "HFC", "Unknown"
        }
        if v not in allowed:
            log.warning(f"Unknown entity type: {v}, defaulting to 'Unknown'")
            return "Unknown"
        return v

    @validator('asset_size')
    def validate_asset_size(cls, v):
        """Validate asset size category"""
        if v is None:
            return v
        allowed = {"small", "medium", "large", "very_large"}
        if v.lower() not in allowed:
            return "medium"  # default
        return v.lower()

    def update_from_query(self, query: str, detected_topics: List[str]):
        """
        Update profile based on query content and detected topics.

        Infers entity details and compliance areas from user queries.
        """
        query_lower = query.lower()

        # Infer entity type from query
        if "we are an nbfc" in query_lower or "our nbfc" in query_lower:
            self.entity_type = "NBFC"
        elif "we are a bank" in query_lower or "our bank" in query_lower:
            self.entity_type = "Bank"
        elif "we are a fintech" in query_lower:
            self.entity_type = "Fintech"
        elif "we are an lsp" in query_lower or "we are a lending service provider" in query_lower:
            self.entity_type = "LSP"

        # Infer license category
        if "nbfc-icc" in query_lower:
            self.license_category = "NBFC-ICC"
        elif "nbfc-mfi" in query_lower:
            self.license_category = "NBFC-MFI"
        elif "nbfc-p2p" in query_lower:
            self.license_category = "NBFC-P2P"

        # Infer products from query
        if "gold loan" in query_lower and "gold_loans" not in self.products:
            self.products.append("gold_loans")
        if "personal loan" in query_lower and "personal_loans" not in self.products:
            self.products.append("personal_loans")
        if "microfinance" in query_lower or "mfi" in query_lower:
            if "microfinance" not in self.products:
                self.products.append("microfinance")
        if "housing loan" in query_lower or "home loan" in query_lower:
            if "housing_loans" not in self.products:
                self.products.append("housing_loans")

        # Infer digital lending
        if any(word in query_lower for word in ["digital lending", "dlg", "fldg", "lsp"]):
            self.digital_lending = True

        if "fldg" in query_lower or "dlg" in query_lower or "first loss" in query_lower:
            self.has_fldg_arrangements = True

        if "lsp" in query_lower or "lending service provider" in query_lower:
            self.uses_lsp = True

        # Infer geography
        if "rural" in query_lower and "rural" not in self.geography:
            self.geography.append("rural")
        if "urban" in query_lower and "urban" not in self.geography:
            self.geography.append("urban")

        # Update compliance areas from detected topics
        topic_to_compliance = {
            "DLG_Cap": "digital_lending",
            "DLG_Eligibility": "digital_lending",
            "Digital_Lending_Core": "digital_lending",
            "Gold_Loan_LTV": "gold_loans",
            "KYC_Process": "kyc_aml",
            "AML_Compliance": "kyc_aml",
            "ECL_Overview": "provisioning",
            "IRACP_Classification": "asset_classification",
            "Model_Governance_Framework": "model_risk",
            "Outsourcing_Risk_Governance": "outsourcing"
        }

        for topic in detected_topics:
            compliance_area = topic_to_compliance.get(topic)
            if compliance_area and compliance_area not in self.compliance_areas:
                self.compliance_areas.append(compliance_area)

        # Update metadata
        self.query_count += 1
        self.updated_at = datetime.now()

    def to_context_string(self) -> str:
        """
        Generate a human-readable context string for LLM prompts.
        """
        parts = []

        if self.entity_name:
            parts.append(f"Entity: {self.entity_name}")

        if self.entity_type and self.entity_type != "Unknown":
            type_str = self.entity_type
            if self.license_category:
                type_str += f" ({self.license_category})"
            parts.append(f"Type: {type_str}")

        if self.asset_size:
            parts.append(f"Size: {self.asset_size}")

        if self.products:
            parts.append(f"Products: {', '.join(self.products)}")

        if self.digital_lending:
            dl_details = []
            if self.has_fldg_arrangements:
                dl_details.append("uses FLDG")
            if self.uses_lsp:
                dl_details.append("uses LSP")
            parts.append(f"Digital Lending: Yes ({', '.join(dl_details)})" if dl_details else "Digital Lending: Yes")

        if self.geography:
            parts.append(f"Geography: {', '.join(self.geography)}")

        if self.compliance_areas:
            parts.append(f"Focus Areas: {', '.join(self.compliance_areas)}")

        return " | ".join(parts) if parts else "No profile information available"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ================================================================================
# PROFILE STORAGE (Redis + JSON Fallback)
# ================================================================================

class ProfileStore:
    """
    Profile storage with Redis primary + JSON file fallback.

    Gracefully falls back to file-based storage if Redis unavailable.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        fallback_dir: str = "data/user_profiles"
    ):
        self.redis_client = None
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        # Try to connect to Redis
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                log.info("✓ Connected to Redis for profile storage")
            except Exception as e:
                log.warning(f"Redis connection failed: {e}. Using file-based fallback.")
                self.redis_client = None
        else:
            log.info("No Redis URL provided. Using file-based profile storage.")

    def save(self, profile: UserProfile) -> bool:
        """Save user profile"""
        try:
            profile_json = profile.json()

            if self.redis_client:
                # Save to Redis with 30-day TTL
                key = f"profile:{profile.user_id}"
                self.redis_client.setex(key, 30 * 24 * 60 * 60, profile_json)
                log.debug(f"Saved profile to Redis: {profile.user_id}")
            else:
                # Fallback to JSON file
                file_path = self.fallback_dir / f"{profile.user_id}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(profile_json)
                log.debug(f"Saved profile to file: {file_path}")

            return True
        except Exception as e:
            log.error(f"Failed to save profile {profile.user_id}: {e}")
            return False

    def load(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile"""
        try:
            if self.redis_client:
                # Try Redis first
                key = f"profile:{user_id}"
                profile_json = self.redis_client.get(key)
                if profile_json:
                    log.debug(f"Loaded profile from Redis: {user_id}")
                    return UserProfile.parse_raw(profile_json)

            # Fallback to file
            file_path = self.fallback_dir / f"{user_id}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    profile_json = f.read()
                log.debug(f"Loaded profile from file: {user_id}")
                return UserProfile.parse_raw(profile_json)

            return None
        except Exception as e:
            log.error(f"Failed to load profile {user_id}: {e}")
            return None

    def delete(self, user_id: str) -> bool:
        """Delete user profile"""
        try:
            if self.redis_client:
                key = f"profile:{user_id}"
                self.redis_client.delete(key)

            file_path = self.fallback_dir / f"{user_id}.json"
            if file_path.exists():
                file_path.unlink()

            log.info(f"Deleted profile: {user_id}")
            return True
        except Exception as e:
            log.error(f"Failed to delete profile {user_id}: {e}")
            return False

    def list_all(self) -> List[str]:
        """List all user IDs"""
        user_ids = []

        try:
            if self.redis_client:
                keys = self.redis_client.keys("profile:*")
                user_ids = [k.replace("profile:", "") for k in keys]
            else:
                user_ids = [f.stem for f in self.fallback_dir.glob("*.json")]
        except Exception as e:
            log.error(f"Failed to list profiles: {e}")

        return user_ids


# ================================================================================
# PROFILE MANAGER (High-level API)
# ================================================================================

class ProfileManager:
    """
    High-level profile management with automatic inference and updates.
    """

    def __init__(self, store: ProfileStore):
        self.store = store

    def get_or_create(self, user_id: str, **kwargs) -> UserProfile:
        """
        Get existing profile or create new one.
        """
        profile = self.store.load(user_id)

        if profile is None:
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                entity_type=kwargs.get("entity_type", "Unknown"),
                entity_name=kwargs.get("entity_name"),
                **kwargs
            )
            self.store.save(profile)
            log.info(f"Created new profile for user: {user_id}")

        return profile

    def update_from_interaction(
        self,
        user_id: str,
        query: str,
        detected_topics: List[str]
    ) -> UserProfile:
        """
        Update profile based on user query and detected topics.
        """
        profile = self.get_or_create(user_id)
        profile.update_from_query(query, detected_topics)
        self.store.save(profile)

        return profile

    def update_fields(self, user_id: str, **fields) -> Optional[UserProfile]:
        """
        Manually update specific profile fields.
        """
        profile = self.store.load(user_id)
        if not profile:
            return None

        for key, value in fields.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.updated_at = datetime.now()
        self.store.save(profile)

        return profile

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.store.load(user_id)

    def delete_profile(self, user_id: str) -> bool:
        """Delete user profile"""
        return self.store.delete(user_id)


# ================================================================================
# GLOBAL INSTANCE (Singleton)
# ================================================================================

_profile_manager = None

def get_profile_manager(redis_url: Optional[str] = None) -> ProfileManager:
    """
    Get global ProfileManager instance.
    """
    global _profile_manager

    if _profile_manager is None:
        store = ProfileStore(redis_url=redis_url)
        _profile_manager = ProfileManager(store)

    return _profile_manager
