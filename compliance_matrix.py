"""
compliance_matrix.py — Compliance Requirement Matrix

Maps Entity Types → Applicable Regulations → Specific Requirements
Supports threshold-based compliance rules based on asset size and products

Author: Finvij Team
Phase: 3 (Days 13-15)
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from user_profile import UserProfile


# ================================================================================
# COMPLIANCE MATRIX DATA STRUCTURE
# ================================================================================

# Regulatory requirements mapped to entity types and products
COMPLIANCE_MATRIX = {
    "NBFC": {
        "gold_loans": [
            "Gold_Loan_LTV",
            "Gold_Loan_Operational"
        ],
        "digital_lending": [
            "DLG_Cap",
            "DLG_Eligibility",
            "DLG_Forms",
            "DLG_Structure",
            "DLG_Restrictions",
            "DLG_Reporting",
            "Digital_Lending_Core",
            "FAQs_Digital_Lending",
            "KFS_Requirements"
        ],
        "microfinance": [
            "IRACP_Classification",
            "IRACP_Provisioning",
            "KYC_Process"
        ],
        "all": [  # Applies to all NBFCs regardless of product
            "KYC_Process",
            "AML_Compliance",
            "IRACP_Classification",
            "IRACP_Provisioning",
            "ECL_Overview",
            "ECL_Stages",
            "ECL_Measurement",
            "Model_Governance_Framework",
            "Model_Governance_Validation"
        ]
    },

    "Bank": {
        "digital_lending": [
            "DLG_Cap",
            "DLG_Eligibility",
            "Digital_Lending_Core",
            "FAQs_Digital_Lending"
        ],
        "gold_loans": [
            "Gold_Loan_LTV",
            "Gold_Loan_Operational"
        ],
        "all": [
            "KYC_Process",
            "AML_Compliance",
            "FAQs_KYC",
            "IRACP_Classification",
            "IRACP_Provisioning",
            "ECL_Overview",
            "ECL_Stages",
            "ECL_Measurement",
            "Model_Governance_Framework",
            "Model_Governance_Validation",
            "Outsourcing_Risk_Governance",
            "Outsourcing_Due_Diligence",
            "Outsourcing_Materiality",
            "Outsourcing_Contractual_Controls",
            "Outsourcing_Monitoring_Audit",
            "Outsourcing_Exit_Strategy",
            "Outsourcing_Directions_2025"
        ]
    },

    "Fintech": {
        "digital_lending": [
            "Digital_Lending_Core",
            "DLG_Cap",
            "DLG_Eligibility",
            "DLG_Restrictions",
            "FAQs_Digital_Lending",
            "KFS_Requirements"
        ],
        "all": [
            "KYC_Process",
            "AML_Compliance",
            "Model_Governance_Framework"
        ]
    },

    "LSP": {
        "all": [
            "Digital_Lending_Core",
            "DLG_Restrictions",
            "FAQs_Digital_Lending",
            "KYC_Process",
            "AML_Compliance",
            "Outsourcing_Due_Diligence"
        ]
    },

    "Cooperative_Bank": {
        "all": [
            "KYC_Process",
            "AML_Compliance",
            "FAQs_KYC",
            "IRACP_Classification",
            "IRACP_Provisioning",
            "ECL_Overview"
        ]
    },

    "Small_Finance_Bank": {
        "all": [
            "KYC_Process",
            "AML_Compliance",
            "IRACP_Classification",
            "IRACP_Provisioning",
            "ECL_Overview",
            "ECL_Stages",
            "Model_Governance_Framework"
        ],
        "digital_lending": [
            "Digital_Lending_Core",
            "DLG_Cap",
            "FAQs_Digital_Lending"
        ]
    },

    "HFC": {
        "all": [
            "KYC_Process",
            "AML_Compliance",
            "IRACP_Classification",
            "IRACP_Provisioning",
            "ECL_Overview",
            "Model_Governance_Framework"
        ],
        "housing_loans": [
            "Gold_Loan_LTV"  # LTV concepts apply to housing loans too
        ]
    }
}


# ================================================================================
# ASSET SIZE THRESHOLDS & SPECIAL RULES
# ================================================================================

ASSET_SIZE_THRESHOLDS = {
    "systemically_important_nbfc": {
        "threshold_crores": 500,
        "applies_to": ["NBFC"],
        "additional_requirements": [
            "Model_Governance_Validation",
            "Outsourcing_Risk_Governance"
        ]
    },

    "deposit_taking_nbfc": {
        "additional_requirements": [
            "IRACP_Provisioning",
            "ECL_Measurement"
        ]
    },

    "large_bank": {
        "threshold_crores": 5000,
        "applies_to": ["Bank"],
        "additional_requirements": [
            "Model_Governance_Validation",
            "Outsourcing_Monitoring_Audit"
        ]
    }
}


# Specific product-based compliance rules
PRODUCT_SPECIFIC_RULES = {
    "gold_loans": {
        "ltv_limit": "75% for non-agriculture, 75% for agriculture",
        "topics": ["Gold_Loan_LTV", "Gold_Loan_Operational"],
        "priority": "high"
    },

    "digital_lending": {
        "fldg_cap": "5% for NBFCs on individual loans, no cap on portfolio",
        "topics": ["DLG_Cap", "DLG_Eligibility", "Digital_Lending_Core"],
        "priority": "high"
    },

    "microfinance": {
        "topics": ["KYC_Process", "IRACP_Classification"],
        "priority": "medium"
    }
}


# ================================================================================
# COMPLIANCE REQUIREMENT MAPPER
# ================================================================================

@dataclass
class ComplianceRequirement:
    """Represents a single compliance requirement"""
    topic_key: str
    topic_label: str
    priority: str  # "immediate", "30_days", "90_days", "ongoing"
    reason: str  # Why this applies to the user
    category: str  # "kyc_aml", "digital_lending", "provisioning", etc.


class ComplianceMatrix:
    """
    Maps user profile to applicable compliance requirements.
    """

    def __init__(self, canonical_topics: Optional[Dict] = None):
        """
        Args:
            canonical_topics: Dict mapping topic keys to labels
        """
        self.canonical_topics = canonical_topics or {}

    def get_applicable_requirements(
        self,
        profile: UserProfile
    ) -> List[ComplianceRequirement]:
        """
        Get all applicable compliance requirements for a user profile.

        Returns:
            List of ComplianceRequirement objects, sorted by priority
        """
        requirements = []
        seen_topics = set()

        # Get entity-specific requirements
        entity_type = profile.entity_type

        if entity_type in COMPLIANCE_MATRIX:
            entity_reqs = COMPLIANCE_MATRIX[entity_type]

            # Add product-specific requirements
            for product in profile.products:
                if product in entity_reqs:
                    for topic_key in entity_reqs[product]:
                        if topic_key not in seen_topics:
                            req = self._create_requirement(
                                topic_key,
                                priority=self._get_priority(topic_key, product),
                                reason=f"Required for {product} product offering",
                                category=self._categorize_topic(topic_key)
                            )
                            requirements.append(req)
                            seen_topics.add(topic_key)

            # Add entity-level "all" requirements
            if "all" in entity_reqs:
                for topic_key in entity_reqs["all"]:
                    if topic_key not in seen_topics:
                        req = self._create_requirement(
                            topic_key,
                            priority=self._get_priority(topic_key, "all"),
                            reason=f"Core requirement for all {entity_type}s",
                            category=self._categorize_topic(topic_key)
                        )
                        requirements.append(req)
                        seen_topics.add(topic_key)

        # Add digital lending specific requirements
        if profile.digital_lending:
            dl_topics = self._get_digital_lending_topics(profile)
            for topic_key in dl_topics:
                if topic_key not in seen_topics:
                    reason = "Required for digital lending operations"
                    if profile.has_fldg_arrangements:
                        reason += " with FLDG/DLG arrangements"
                    if profile.uses_lsp:
                        reason += " using LSP"

                    req = self._create_requirement(
                        topic_key,
                        priority="immediate",
                        reason=reason,
                        category="digital_lending"
                    )
                    requirements.append(req)
                    seen_topics.add(topic_key)

        # Add asset-size based requirements
        if profile.asset_size:
            size_reqs = self._get_size_based_requirements(profile)
            for topic_key in size_reqs:
                if topic_key not in seen_topics:
                    req = self._create_requirement(
                        topic_key,
                        priority="30_days",
                        reason=f"Required for {profile.asset_size}-scale entities",
                        category=self._categorize_topic(topic_key)
                    )
                    requirements.append(req)
                    seen_topics.add(topic_key)

        # Sort by priority
        priority_order = {"immediate": 0, "30_days": 1, "90_days": 2, "ongoing": 3}
        requirements.sort(key=lambda r: priority_order.get(r.priority, 99))

        return requirements

    def _create_requirement(
        self,
        topic_key: str,
        priority: str,
        reason: str,
        category: str
    ) -> ComplianceRequirement:
        """Create a ComplianceRequirement object"""
        topic_label = self.canonical_topics.get(topic_key, {}).get("label", topic_key)

        return ComplianceRequirement(
            topic_key=topic_key,
            topic_label=topic_label,
            priority=priority,
            reason=reason,
            category=category
        )

    def _get_priority(self, topic_key: str, product: str) -> str:
        """Determine priority level for a requirement"""
        # High priority topics
        immediate_topics = {
            "DLG_Cap", "DLG_Eligibility", "Gold_Loan_LTV",
            "KYC_Process", "AML_Compliance", "Digital_Lending_Core"
        }

        if topic_key in immediate_topics:
            return "immediate"

        # Product-specific priorities
        if product == "digital_lending":
            return "immediate"

        if product == "gold_loans" and "Gold_Loan" in topic_key:
            return "immediate"

        # Model governance and validation
        if "Model_Governance" in topic_key:
            return "90_days"

        # ECL and provisioning
        if "ECL" in topic_key or "IRACP" in topic_key:
            return "30_days"

        return "ongoing"

    def _categorize_topic(self, topic_key: str) -> str:
        """Categorize topic into compliance area"""
        if "KYC" in topic_key or "AML" in topic_key:
            return "kyc_aml"
        elif "DLG" in topic_key or "Digital_Lending" in topic_key:
            return "digital_lending"
        elif "Gold_Loan" in topic_key:
            return "gold_loans"
        elif "ECL" in topic_key or "IRACP" in topic_key:
            return "provisioning"
        elif "Model" in topic_key:
            return "model_risk"
        elif "Outsourcing" in topic_key:
            return "outsourcing"
        elif "KFS" in topic_key:
            return "disclosure"
        else:
            return "general"

    def _get_digital_lending_topics(self, profile: UserProfile) -> List[str]:
        """Get digital lending specific topics"""
        topics = [
            "Digital_Lending_Core",
            "FAQs_Digital_Lending",
            "KFS_Requirements"
        ]

        if profile.has_fldg_arrangements:
            topics.extend([
                "DLG_Cap",
                "DLG_Eligibility",
                "DLG_Forms",
                "DLG_Structure",
                "DLG_Restrictions",
                "DLG_Reporting"
            ])

        if profile.uses_lsp:
            topics.extend([
                "DLG_Restrictions",
                "Outsourcing_Due_Diligence"
            ])

        return topics

    def _get_size_based_requirements(self, profile: UserProfile) -> List[str]:
        """Get requirements based on asset size"""
        requirements = []

        # For large/very_large entities, add governance requirements
        if profile.asset_size in ["large", "very_large"]:
            requirements.extend([
                "Model_Governance_Validation",
                "Outsourcing_Monitoring_Audit",
                "ECL_Measurement"
            ])

        return requirements

    def get_requirements_by_category(
        self,
        requirements: List[ComplianceRequirement]
    ) -> Dict[str, List[ComplianceRequirement]]:
        """Group requirements by category"""
        categorized = {}

        for req in requirements:
            if req.category not in categorized:
                categorized[req.category] = []
            categorized[req.category].append(req)

        return categorized

    def get_requirements_by_priority(
        self,
        requirements: List[ComplianceRequirement]
    ) -> Dict[str, List[ComplianceRequirement]]:
        """Group requirements by priority"""
        prioritized = {
            "immediate": [],
            "30_days": [],
            "90_days": [],
            "ongoing": []
        }

        for req in requirements:
            if req.priority in prioritized:
                prioritized[req.priority].append(req)

        return prioritized


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def load_canonical_topics(path: str = "canonical_topics.json") -> Dict:
    """Load canonical topics from JSON file"""
    import json
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load canonical topics: {e}")
        return {}


def get_compliance_matrix(canonical_topics_path: str = "canonical_topics.json") -> ComplianceMatrix:
    """Get initialized compliance matrix"""
    topics = load_canonical_topics(canonical_topics_path)
    return ComplianceMatrix(canonical_topics=topics)
