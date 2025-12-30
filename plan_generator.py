"""
plan_generator.py — Compliance Plan Generator

Generates customized compliance plans with actionable steps based on:
- User profile (entity type, products, scale)
- Applicable regulations from compliance matrix
- Retrieved regulatory content from RAG
- Timeline-based prioritization

Author: Finvij Team
Phase: 3 (Days 16-21)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

from groq import Groq
from user_profile import UserProfile
from compliance_matrix import ComplianceMatrix, ComplianceRequirement, get_compliance_matrix

log = logging.getLogger("plan_generator")


# ================================================================================
# PLAN GENERATION PROMPTS
# ================================================================================

PLAN_GENERATION_PROMPT = """You are an expert RBI compliance advisor creating a customized compliance plan.

USER PROFILE:
- Entity Type: {entity_type}
- License: {license_category}
- Products: {products}
- Asset Size: {asset_size}
- Digital Lending: {digital_lending}
- FLDG Arrangements: {has_fldg}
- Uses LSP: {uses_lsp}
- Geography: {geography}
- Current Compliance Areas: {compliance_areas}

APPLICABLE REGULATIONS:
The following regulatory topics apply to this entity:

{applicable_requirements}

RELEVANT REGULATORY CONTENT:
{regulatory_chunks}

KNOWLEDGE GRAPH FACTS:
{kg_facts}

TASK:
Generate a COMPREHENSIVE, ACTIONABLE compliance plan tailored to this specific entity.

REQUIREMENTS:
1. **Priority Areas** - Identify 3-5 critical compliance areas based on their products and entity type
2. **Specific Requirements** - For each area, list concrete requirements with:
   - Exact RBI circular/guideline references
   - Specific thresholds, limits, or percentages
   - Entity-specific applicability
3. **Timelines** - Categorize actions as:
   - IMMEDIATE (0-15 days): Critical compliance gaps
   - 30 DAYS: Important setup/documentation
   - 90 DAYS: Process improvement and validation
   - ONGOING: Continuous compliance and monitoring
4. **Action Items** - For each requirement, provide:
   - Specific action to take
   - Responsible party (Board/Senior Management/Compliance/Operations/Risk)
   - Expected deliverable
   - Success criteria
5. **Risk Areas** - Identify 2-3 high-risk areas specific to their entity type and products
6. **Implementation Checklist** - Provide a step-by-step checklist for each priority area

FORMAT AS STRUCTURED JSON:
{{
  "summary": "Brief 2-3 sentence overview of compliance status and plan",
  "priority_areas": [
    {{
      "area": "Area name",
      "criticality": "high/medium/low",
      "reason": "Why this is a priority for this entity"
    }}
  ],
  "timeline_based_actions": {{
    "immediate": [
      {{
        "requirement": "Specific requirement",
        "action": "Concrete action to take",
        "responsible": "Responsible party",
        "deliverable": "Expected output",
        "rbi_reference": "Circular/guideline reference"
      }}
    ],
    "30_days": [...],
    "90_days": [...],
    "ongoing": [...]
  }},
  "risk_areas": [
    {{
      "risk": "Risk description",
      "impact": "Potential impact",
      "mitigation": "How to mitigate"
    }}
  ],
  "implementation_checklist": {{
    "area_name": [
      "Step 1",
      "Step 2",
      ...
    ]
  }}
}}

IMPORTANT:
- Be SPECIFIC, not generic
- Use EXACT numbers (e.g., "5% FLDG cap", "75% LTV limit")
- Reference ACTUAL RBI circulars and guidelines
- Tailor everything to THIS specific entity type and products
- Do NOT provide generic advice that applies to everyone
- Return ONLY valid JSON, no markdown, no extra text
"""


PLAN_REFINEMENT_PROMPT = """You are refining a compliance plan based on user feedback.

ORIGINAL PLAN:
{original_plan}

USER FEEDBACK:
{user_feedback}

TASK:
Update the compliance plan based on the user's feedback. Maintain the same JSON structure but modify the content according to their request.

Examples of refinements:
- "Focus more on digital lending" → Expand digital lending sections, add more DLG/FLDG details
- "Add more detail on KYC" → Expand KYC requirements with step-by-step procedures
- "Prioritize gold loan compliance" → Move gold loan items to immediate timeline
- "We don't do microfinance" → Remove microfinance-related items

Return ONLY the updated JSON plan, no markdown, no extra text.
"""


# ================================================================================
# PLAN DATA STRUCTURES
# ================================================================================

@dataclass
class ActionItem:
    """Single action item in compliance plan"""
    requirement: str
    action: str
    responsible: str
    deliverable: str
    rbi_reference: str


@dataclass
class PriorityArea:
    """Priority compliance area"""
    area: str
    criticality: str
    reason: str


@dataclass
class RiskArea:
    """Risk area"""
    risk: str
    impact: str
    mitigation: str


@dataclass
class CompliancePlan:
    """Complete compliance plan"""
    plan_id: str
    user_id: str
    generated_at: datetime
    version: int

    # Profile snapshot
    entity_type: str
    products: List[str]

    # Plan content
    summary: str
    priority_areas: List[PriorityArea]
    timeline_based_actions: Dict[str, List[ActionItem]]
    risk_areas: List[RiskArea]
    implementation_checklist: Dict[str, List[str]]

    # Metadata
    requirements_count: int
    applicable_topics: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "generated_at": self.generated_at.isoformat(),
            "version": self.version,
            "entity_type": self.entity_type,
            "products": self.products,
            "summary": self.summary,
            "priority_areas": [asdict(pa) for pa in self.priority_areas],
            "timeline_based_actions": {
                timeline: [asdict(ai) for ai in actions]
                for timeline, actions in self.timeline_based_actions.items()
            },
            "risk_areas": [asdict(ra) for ra in self.risk_areas],
            "implementation_checklist": self.implementation_checklist,
            "requirements_count": self.requirements_count,
            "applicable_topics": self.applicable_topics
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


# ================================================================================
# PLAN GENERATOR CLASS
# ================================================================================

class PlanGenerator:
    """
    Generates customized compliance plans for financial entities.
    """

    def __init__(
        self,
        groq_client: Groq,
        compliance_matrix: ComplianceMatrix,
        llm_model: str = "llama-3.3-70b-versatile"
    ):
        self.groq = groq_client
        self.compliance_matrix = compliance_matrix
        self.llm_model = llm_model

    def generate_plan(
        self,
        profile: UserProfile,
        chunks: List[Tuple[str, str]],
        kg_facts: List[Dict],
        plan_version: int = 1
    ) -> CompliancePlan:
        """
        Generate a complete compliance plan for a user.

        Args:
            profile: User profile
            chunks: Retrieved regulatory chunks
            kg_facts: Knowledge graph facts
            plan_version: Version number for plan

        Returns:
            CompliancePlan object
        """
        log.info(f"Generating compliance plan for user {profile.user_id}, version {plan_version}")

        # Step 1: Get applicable requirements
        requirements = self.compliance_matrix.get_applicable_requirements(profile)
        log.info(f"Found {len(requirements)} applicable requirements")

        # Step 2: Format requirements for prompt
        requirements_text = self._format_requirements(requirements)

        # Step 3: Format chunks and KG facts
        chunks_text = self._format_chunks(chunks)
        kg_text = self._format_kg_facts(kg_facts)

        # Step 4: Build prompt
        prompt = self._build_prompt(profile, requirements_text, chunks_text, kg_text)

        # Step 5: Call LLM
        try:
            response = self.groq.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent plans
                max_tokens=4000
            )

            plan_json_raw = response.choices[0].message.content.strip()

            # Clean up potential markdown
            if plan_json_raw.startswith("```"):
                plan_json_raw = plan_json_raw.strip("`").strip()
                if plan_json_raw.lower().startswith("json"):
                    plan_json_raw = plan_json_raw[4:].strip()

            plan_data = json.loads(plan_json_raw)

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse plan JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON: {plan_json_raw[:200]}...")
        except Exception as e:
            log.error(f"LLM call failed: {e}")
            raise

        # Step 6: Parse into CompliancePlan object
        plan = self._parse_plan(profile, plan_data, requirements, plan_version)

        log.info(f"Successfully generated plan {plan.plan_id} with {plan.requirements_count} requirements")
        return plan

    def refine_plan(
        self,
        original_plan: CompliancePlan,
        user_feedback: str,
        profile: UserProfile
    ) -> CompliancePlan:
        """
        Refine an existing plan based on user feedback.

        Args:
            original_plan: Original compliance plan
            user_feedback: User's refinement request
            profile: User profile

        Returns:
            New CompliancePlan with incremented version
        """
        log.info(f"Refining plan {original_plan.plan_id} based on feedback: {user_feedback[:100]}")

        prompt = PLAN_REFINEMENT_PROMPT.format(
            original_plan=original_plan.to_json(),
            user_feedback=user_feedback
        )

        try:
            response = self.groq.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )

            plan_json_raw = response.choices[0].message.content.strip()

            # Clean up markdown
            if plan_json_raw.startswith("```"):
                plan_json_raw = plan_json_raw.strip("`").strip()
                if plan_json_raw.lower().startswith("json"):
                    plan_json_raw = plan_json_raw[4:].strip()

            plan_data = json.loads(plan_json_raw)

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse refined plan JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON during refinement")
        except Exception as e:
            log.error(f"Plan refinement failed: {e}")
            raise

        # Parse into new plan with incremented version
        requirements = []  # Empty for refined plans
        new_version = original_plan.version + 1
        refined_plan = self._parse_plan(profile, plan_data, requirements, new_version)

        log.info(f"Successfully refined plan to version {new_version}")
        return refined_plan

    def _build_prompt(
        self,
        profile: UserProfile,
        requirements_text: str,
        chunks_text: str,
        kg_text: str
    ) -> str:
        """Build the plan generation prompt"""
        return PLAN_GENERATION_PROMPT.format(
            entity_type=profile.entity_type or "Unknown",
            license_category=profile.license_category or "Not specified",
            products=", ".join(profile.products) if profile.products else "Not specified",
            asset_size=profile.asset_size or "Not specified",
            digital_lending="Yes" if profile.digital_lending else "No",
            has_fldg="Yes" if profile.has_fldg_arrangements else "No",
            uses_lsp="Yes" if profile.uses_lsp else "No",
            geography=", ".join(profile.geography) if profile.geography else "Not specified",
            compliance_areas=", ".join(profile.compliance_areas) if profile.compliance_areas else "None detected",
            applicable_requirements=requirements_text,
            regulatory_chunks=chunks_text,
            kg_facts=kg_text
        )

    def _format_requirements(self, requirements: List[ComplianceRequirement]) -> str:
        """Format requirements for prompt"""
        if not requirements:
            return "No specific requirements identified."

        lines = []
        for i, req in enumerate(requirements, 1):
            lines.append(
                f"{i}. [{req.priority.upper()}] {req.topic_label}\n"
                f"   Category: {req.category}\n"
                f"   Reason: {req.reason}"
            )

        return "\n\n".join(lines)

    def _format_chunks(self, chunks: List[Tuple[str, str]]) -> str:
        """Format regulatory chunks for prompt"""
        if not chunks:
            return "No regulatory content retrieved."

        lines = []
        for i, (chunk_id, text) in enumerate(chunks[:15], 1):  # Limit to 15 chunks
            lines.append(f"[Chunk {i}]\n{text[:800]}")  # Limit chunk length

        return "\n\n".join(lines)

    def _format_kg_facts(self, kg_facts: List[Dict]) -> str:
        """Format KG facts for prompt"""
        if not kg_facts:
            return "No knowledge graph facts available."

        lines = []
        for i, fact in enumerate(kg_facts[:20], 1):  # Limit to 20 facts
            head = fact.get("head", "")
            rel = fact.get("relation", "")
            tail = fact.get("tail", "")
            lines.append(f"{i}. {head} → {rel} → {tail}")

        return "\n".join(lines)

    def _parse_plan(
        self,
        profile: UserProfile,
        plan_data: Dict,
        requirements: List[ComplianceRequirement],
        version: int
    ) -> CompliancePlan:
        """Parse plan data into CompliancePlan object"""
        import uuid

        # Parse priority areas
        priority_areas = [
            PriorityArea(**pa) for pa in plan_data.get("priority_areas", [])
        ]

        # Parse timeline-based actions
        timeline_actions = {}
        for timeline, actions in plan_data.get("timeline_based_actions", {}).items():
            timeline_actions[timeline] = [
                ActionItem(**action) for action in actions
            ]

        # Parse risk areas
        risk_areas = [
            RiskArea(**ra) for ra in plan_data.get("risk_areas", [])
        ]

        # Extract applicable topics
        applicable_topics = [req.topic_key for req in requirements]

        return CompliancePlan(
            plan_id=str(uuid.uuid4())[:8],
            user_id=profile.user_id,
            generated_at=datetime.now(),
            version=version,
            entity_type=profile.entity_type,
            products=profile.products,
            summary=plan_data.get("summary", ""),
            priority_areas=priority_areas,
            timeline_based_actions=timeline_actions,
            risk_areas=risk_areas,
            implementation_checklist=plan_data.get("implementation_checklist", {}),
            requirements_count=len(requirements),
            applicable_topics=applicable_topics
        )


# ================================================================================
# PLAN STORAGE
# ================================================================================

class PlanStore:
    """
    Store compliance plans in JSON files.
    """

    def __init__(self, storage_dir: str = "data/compliance_plans"):
        from pathlib import Path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, plan: CompliancePlan) -> bool:
        """Save plan to file"""
        try:
            filename = f"{plan.user_id}_v{plan.version}_{plan.plan_id}.json"
            filepath = self.storage_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(plan.to_json())

            log.info(f"Saved plan to {filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to save plan: {e}")
            return False

    def load(self, user_id: str, version: Optional[int] = None) -> Optional[CompliancePlan]:
        """Load plan from file"""
        try:
            if version:
                # Load specific version
                pattern = f"{user_id}_v{version}_*.json"
            else:
                # Load latest version
                pattern = f"{user_id}_v*_*.json"

            matching_files = list(self.storage_dir.glob(pattern))
            if not matching_files:
                return None

            # Get the latest file
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

            with open(latest_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)

            # Reconstruct CompliancePlan (simplified - you may need to enhance this)
            return plan_data  # Return dict for now

        except Exception as e:
            log.error(f"Failed to load plan: {e}")
            return None

    def list_plans(self, user_id: str) -> List[Dict]:
        """List all plans for a user"""
        try:
            pattern = f"{user_id}_v*_*.json"
            files = self.storage_dir.glob(pattern)

            plans = []
            for filepath in files:
                with open(filepath, "r", encoding="utf-8") as f:
                    plan_data = json.load(f)
                    plans.append({
                        "plan_id": plan_data.get("plan_id"),
                        "version": plan_data.get("version"),
                        "generated_at": plan_data.get("generated_at"),
                        "summary": plan_data.get("summary", "")[:100]
                    })

            return sorted(plans, key=lambda p: p["version"], reverse=True)

        except Exception as e:
            log.error(f"Failed to list plans: {e}")
            return []
