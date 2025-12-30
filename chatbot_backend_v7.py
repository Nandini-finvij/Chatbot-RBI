"""
chatbot_backend_v7.py — With User Profiling & Compliance Plan Generation (Phase 3 Complete)

Phase 2 Features:
✅ User profile management
✅ Profile-aware retrieval & re-ranking
✅ Onboarding conversation flow
✅ Personalized responses
✅ Profile commands (/profile, /setup)

Phase 3 Features (NEW):
✅ Compliance plan generation
✅ Entity-specific regulatory mapping
✅ Timeline-based action plans
✅ Plan versioning & refinement
✅ Multi-format output (JSON, Markdown, PDF, HTML)

Author: Finvij Team
Version: 8.0.0 (Phase 3)
"""

import os
import uuid
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import torch

# Phase 1 imports
from hybrid_retrieval import hybrid_retrieve, rag_search
from prompt_builder import build_prompt
from dynamic_topic_matcher import get_topic_matcher

# Phase 2 imports
from user_profile import UserProfile, get_profile_manager
from profile_aware_retrieval import profile_aware_retrieve, build_profile_context
from onboarding_flow import get_onboarding_handler, OnboardingState

# Phase 3 imports
from compliance_matrix import get_compliance_matrix, ComplianceMatrix
from plan_generator import PlanGenerator, PlanStore
from plan_formatter import PlanFormatter

# Readability measurement
from readability_analyzer import get_readability_analyzer, ReadabilityScore

# ================================================================================
# CONFIGURATION
# ================================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
CANONICAL_TOPICS_PATH = os.environ.get("CANONICAL_TOPICS", "canonical_topics.json")
TOPIC_EMBEDDINGS_PATH = os.environ.get("TOPIC_EMBEDDINGS", "data/embeddings/topic_embeddings.pt")

# NEW: Profile settings
REDIS_URL = os.environ.get("REDIS_URL", None)  # Optional: redis://localhost:6379
PROFILE_RERANK_ALPHA = float(os.environ.get("PROFILE_RERANK_ALPHA", "0.4"))  # 40% profile weight

# Thresholds
TOPIC_MATCH_THRESHOLD = 0.55
TOPIC_MARGINAL_THRESHOLD = 0.40
GREETING_THRESHOLD = 0.75
FOLLOWUP_THRESHOLD = 0.60
FOLLOWUP_PATTERN_THRESHOLD = 0.65

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("rbi-chatbot-v7")

# ================================================================================
# INITIALIZATION
# ================================================================================

# Initialize LLM client
groq = Groq(api_key=GROQ_API_KEY)
log.info("✓ Groq client initialized")

# Initialize embedder
embedder = SentenceTransformer(EMBED_MODEL)
log.info(f"✓ Embedder loaded: {EMBED_MODEL}")

# Initialize topic matcher
topic_matcher = get_topic_matcher(
    canonical_topics_path=CANONICAL_TOPICS_PATH,
    embeddings_path=TOPIC_EMBEDDINGS_PATH,
    model=embedder,
    similarity_threshold=TOPIC_MATCH_THRESHOLD
)

if not os.path.exists(TOPIC_EMBEDDINGS_PATH):
    log.warning("⚠ Topic embeddings not found. Building now...")
    topic_matcher.build_embeddings(save=True)

log.info(f"✓ Topic matcher initialized with {len(topic_matcher.get_all_topics())} topics")

# NEW: Initialize profile manager
profile_manager = get_profile_manager(redis_url=REDIS_URL)
log.info("✓ Profile manager initialized")

# NEW: Initialize onboarding handler
onboarding_handler = get_onboarding_handler()
log.info("✓ Onboarding handler initialized")

# Phase 3: Initialize compliance matrix and plan generator
compliance_matrix = get_compliance_matrix(CANONICAL_TOPICS_PATH)
log.info("✓ Compliance matrix initialized")

plan_generator = PlanGenerator(
    groq_client=groq,
    compliance_matrix=compliance_matrix,
    llm_model=LLM_MODEL
)
log.info("✓ Plan generator initialized")

plan_store = PlanStore(storage_dir="data/compliance_plans")
log.info("✓ Plan store initialized")

# Initialize readability analyzer
readability_analyzer = get_readability_analyzer()
log.info("✓ Readability analyzer initialized")

# ================================================================================
# SESSION MEMORY
# ================================================================================

memory: Dict[str, Dict[str, Any]] = {}

def get_session(cid: str) -> Dict[str, Any]:
    if cid not in memory:
        memory[cid] = {
            "prev_answer": None,
            "prev_query": None,
            "prev_topics": [],
            "turn_count": 0,
            "created_at": None
        }
    return memory[cid]

def update_session(cid: str, query: str, answer: str, topics: List[str]):
    session = get_session(cid)
    session["prev_answer"] = answer
    session["prev_query"] = query
    session["prev_topics"] = topics
    session["turn_count"] += 1

def clear_session(cid: str):
    memory.pop(cid, None)
    log.info(f"Session cleared: {cid}")

# ================================================================================
# GREETING & FOLLOW-UP DETECTION (Same as V6)
# ================================================================================

GREETING_EXAMPLES = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "hi there", "hello there", "hey there",
    "greetings", "howdy", "what's up", "sup", "yo"
]
GREETING_EMBEDDINGS = embedder.encode(GREETING_EXAMPLES, convert_to_tensor=True)

def is_greeting(query: str) -> bool:
    if len(query.split()) > 5:
        return False
    query_emb = embedder.encode(query.lower(), convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, GREETING_EMBEDDINGS)
    return similarities.max().item() >= GREETING_THRESHOLD

FOLLOWUP_INDICATORS = [
    "tell me more", "explain more", "more details", "elaborate",
    "what about", "and what", "also", "additionally",
    "can you clarify", "clarify that", "explain that",
    "what do you mean", "meaning of", "define",
    "why is that", "how does that", "when does",
    "previous answer", "you mentioned", "you said",
    "regarding that", "about that", "on that",
    "more info", "further details", "continue",
    "again", "repeat", "say that again",
    "more about", "tell me about", "what else",
    "give me more", "provide more", "expand on",
    "go on", "keep going", "more on that",
    "further", "expand", "detail"
]
FOLLOWUP_EMBEDDINGS = embedder.encode(FOLLOWUP_INDICATORS, convert_to_tensor=True)

def is_followup(query: str, session: Dict[str, Any]) -> bool:
    """
    Detect if query is a follow-up to previous conversation.

    Uses multiple signals:
    1. Pattern matching (e.g., "tell me more", "what about")
    2. Semantic similarity to previous answer
    3. Short query with low topic match (likely contextual)
    4. Previous topic relevance
    """
    # No previous context
    if not session.get("prev_answer") or session.get("turn_count", 0) == 0:
        return False

    query_lower = query.lower().strip()
    word_count = len(query.split())

    # Signal 1: Direct follow-up pattern match
    query_emb = embedder.encode(query_lower, convert_to_tensor=True)
    pattern_similarities = util.cos_sim(query_emb, FOLLOWUP_EMBEDDINGS)
    max_pattern_sim = pattern_similarities.max().item()

    if max_pattern_sim >= FOLLOWUP_PATTERN_THRESHOLD:
        log.info(f"Follow-up detected (pattern match: {max_pattern_sim:.3f})")
        return True

    # Signal 2: Semantic similarity to previous answer (for short queries)
    prev_answer = session.get("prev_answer", "")
    if prev_answer and word_count <= 10:
        prev_emb = embedder.encode(prev_answer[:500], convert_to_tensor=True)
        context_sim = util.cos_sim(query_emb, prev_emb).item()
        if context_sim >= FOLLOWUP_THRESHOLD:
            log.info(f"Follow-up detected (context similarity: {context_sim:.3f})")
            return True

    # Signal 3: Check similarity to previous query
    prev_query = session.get("prev_query", "")
    if prev_query and word_count <= 8:
        prev_query_emb = embedder.encode(prev_query.lower(), convert_to_tensor=True)
        query_sim = util.cos_sim(query_emb, prev_query_emb).item()
        if query_sim >= 0.7:  # Very similar to previous query
            log.info(f"Follow-up detected (similar to prev query: {query_sim:.3f})")
            return True

    # Signal 4: Short query with weak topic match (likely needs context)
    if word_count <= 6:
        topic_result = topic_matcher.match(query)
        if topic_result:
            _, score = topic_result
            if score < 0.6:  # Low topic confidence, likely needs context
                log.info(f"Follow-up detected (short query, low topic match: {score:.3f})")
                return True

    # Signal 5: Check if query references previous topics
    prev_topics = session.get("prev_topics", [])
    if prev_topics and word_count <= 8:
        # If query mentions terms from previous topics, likely a follow-up
        for topic in prev_topics:
            if topic.lower().replace("_", " ") in query_lower:
                log.info(f"Follow-up detected (references prev topic: {topic})")
                return True

    return False

def rewrite_followup(query: str, session: Dict[str, Any]) -> str:
    prev_query = session.get("prev_query", "")
    prev_answer = session.get("prev_answer", "")[:500]

    prompt = f"""Rewrite the follow-up question to be a complete, self-contained question about RBI regulations.

Previous Question: {prev_query}
Previous Answer: {prev_answer}
Follow-up Question: {query}

Task: Rewrite the follow-up as a complete question that includes necessary context from the previous conversation.

Rules:
- Keep it focused on RBI regulations
- Include relevant context from previous Q&A
- Make it understandable without seeing the previous conversation
- Return ONLY the rewritten question, nothing else

Rewritten Question:"""

    try:
        response = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Follow-up rewrite failed: {e}")
        return query

# ================================================================================
# INTENT CLASSIFICATION
# ================================================================================

def classify_intent(query: str, session: Dict[str, Any]) -> Tuple[str, List[Tuple[str, float]]]:
    # Check for profile setup command
    query_lower = query.lower().strip()
    if query_lower in ("/setup", "/onboard", "setup profile", "setup my profile"):
        return "setup_profile", []

    if query_lower in ("/profile", "show profile", "my profile", "view profile"):
        return "view_profile", []

    # Phase 3: Check for plan generation commands
    if query_lower in ("/plan", "/generate-plan", "generate plan", "generate compliance plan",
                       "create compliance plan", "generate my compliance plan", "create my plan"):
        return "generate_plan", []

    if query_lower.startswith("/refine-plan") or "refine plan" in query_lower or "update plan" in query_lower:
        return "refine_plan", []

    if query_lower in ("/view-plan", "view plan", "show plan", "show my plan"):
        return "view_plan", []

    # Check for greeting
    if is_greeting(query):
        return "greeting", []

    # Check for follow-up
    if is_followup(query, session):
        prev_topics = session.get("prev_topics", [])
        if prev_topics:
            return "followup", [(t, 1.0) for t in prev_topics]
        topics = topic_matcher.match_multiple(query, top_k=2)
        return "followup", topics

    # Match to RBI topics
    topics = topic_matcher.match_multiple(query, top_k=3)

    if topics and topics[0][1] >= TOPIC_MATCH_THRESHOLD:
        return "rbi_query", topics

    if topics and topics[0][1] >= TOPIC_MARGINAL_THRESHOLD:
        return "rbi_query", topics

    return "out_of_scope", topics

# ================================================================================
# RESPONSE GENERATORS
# ================================================================================

def greeting_response(query: str, profile: Optional[UserProfile] = None) -> str:
    all_topics = topic_matcher.get_all_topics()
    topic_count = len(all_topics)

    greeting = f"""Hello! I'm your RBI regulatory assistant.

I can help you with **{topic_count} regulatory topics** including:
• Digital Lending & FLDG guidelines
• KYC/AML compliance requirements
• Gold loan regulations & LTV limits
• ECL provisioning & asset classification (IRACP)
• Outsourcing risk management
• Model governance & validation
• And many more RBI topics!"""

    # Add personalized note if profile exists
    if profile and profile.entity_type != "Unknown":
        greeting += f"\n\n**Your Profile:** {profile.to_context_string()}"
        greeting += "\n\n💡 _I'll provide answers tailored to your organization._"

    greeting += "\n\n**What would you like to know about RBI regulations?**"

    if not profile or profile.entity_type == "Unknown":
        greeting += "\n\n_Tip: Use `/setup` to create your profile for personalized guidance._"

    return greeting

def out_of_scope_response(query: str, topics: List[Tuple[str, float]]) -> str:
    if topics:
        closest_key, closest_score = topics[0]
        closest_label = topic_matcher.get_topic_label(closest_key)
        return f"""I'm not confident I understand your question in the context of RBI regulations.

The closest topic I found is **{closest_label}** (confidence: {closest_score:.0%}).

Could you rephrase your question to be more specific about RBI regulations?

**For example, you could ask about:**
• FLDG/DLG cap limits and eligibility
• KYC requirements for different entities
• Gold loan LTV limits and operational norms
• ECL provisioning and measurement
• Outsourcing governance and due diligence"""

    return """I'm specialized in RBI regulatory matters and couldn't find a relevant topic for your question.

**I can help with topics like:**
• Digital Lending & FLDG regulations
• KYC/AML compliance
• Gold loan norms
• Asset classification & provisioning (IRACP)
• Outsourcing directions
• Model risk management

**Please ask me anything about RBI regulations!**"""

def generate_answer(
    query: str,
    chunks: List[Tuple[str, str]],
    kg_facts: List[Dict],
    topics: List[Tuple[str, float]],
    profile: Optional[UserProfile] = None
) -> str:
    # Build prompt with profile context
    base_prompt = build_prompt(query, chunks, kg_facts)

    # Add profile context if available
    if profile:
        profile_context = build_profile_context(profile)
        if profile_context:
            base_prompt = f"""USER PROFILE:
{profile_context}

{base_prompt}

NOTE: Tailor your answer to the user's profile (entity type, products, scale) when relevant."""

    try:
        response = groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": base_prompt}],
            temperature=0,
            max_tokens=800
        )
        answer = response.choices[0].message.content.strip()
        log.info(f"Generated answer ({len(answer)} chars) for query: '{query[:50]}...'")
        return answer
    except Exception as e:
        log.error(f"LLM answer generation failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return f"⚠️ I encountered an error generating the response: {str(e)}\n\nPlease try again or rephrase your question."

def analyze_response_readability(answer: str) -> Optional[Dict]:
    """
    Analyze the readability of a chatbot response.

    Returns:
        Dictionary with readability scores or None if analysis fails
    """
    try:
        score = readability_analyzer.analyze(answer)
        return {
            "flesch_reading_ease": round(score.flesch_reading_ease, 2),
            "flesch_kincaid_grade": round(score.flesch_kincaid_grade, 2),
            "readability_level": score.readability_level,
            "grade_level": score.grade_level_interpretation,
            "avg_sentence_length": round(score.avg_sentence_length, 1),
            "complex_word_percentage": round(score.complex_word_percentage, 1),
            "total_words": score.total_words,
            "total_sentences": score.total_sentences
        }
    except Exception as e:
        log.error(f"Readability analysis failed: {e}")
        return None

# ================================================================================
# FASTAPI APPLICATION
# ================================================================================

app = FastAPI(
    title="RBI Regulatory Chatbot V7",
    description="RBI chatbot with user profiling and personalized retrieval (Phase 2 Complete)",
    version="7.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================================
# REQUEST/RESPONSE MODELS
# ================================================================================

class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None  # NEW: for profile tracking
    clear_session: bool = False

class AskResponse(BaseModel):
    conversation_id: str
    user_id: Optional[str] = None
    answer: str
    intent: str
    matched_topics: List[Dict[str, Any]]
    chunks_used: List[Dict[str, str]]
    kg_facts: List[Dict[str, Any]]
    confidence: float
    profile_summary: Optional[str] = None
    onboarding_active: bool = False
    compliance_plan: Optional[Dict] = None  # Phase 3: Compliance plan data
    readability: Optional[Dict] = None  # Readability scores

class PlanRequest(BaseModel):
    user_id: str
    output_format: str = "markdown"  # markdown, json, pdf, html

class PlanRefineRequest(BaseModel):
    user_id: str
    feedback: str
    output_format: str = "markdown"

# ================================================================================
# API ENDPOINTS
# ================================================================================

@app.post("/ask", response_model=AskResponse)
@app.post("/chat")  # Add /chat endpoint alias
def ask(req: AskRequest):
    """Main chat endpoint with profile support"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Session management
    cid = req.conversation_id or str(uuid.uuid4())
    user_id = req.user_id or cid  # Use conversation_id as user_id if not provided

    if req.clear_session:
        clear_session(cid)

    session = get_session(cid)
    query = req.question.strip()

    log.info(f"[{cid[:8]}] Query: '{query}'")

    # Check if onboarding is active
    if onboarding_handler.is_onboarding_active(user_id):
        # Process onboarding response
        result = onboarding_handler.process_response(user_id, query)

        if result.get("completed"):
            # Onboarding completed - create profile
            profile = onboarding_handler.complete_onboarding(user_id)
            message = result.get("message", "Profile setup complete!")

            return AskResponse(
                conversation_id=cid,
                user_id=user_id,
                answer=message,
                intent="onboarding_complete",
                matched_topics=[],
                chunks_used=[],
                kg_facts=[],
                confidence=1.0,
                profile_summary=profile.to_context_string(),
                onboarding_active=False
            )

        # Return next question
        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=result["next_question"],
            intent="onboarding_question",
            matched_topics=[],
            chunks_used=[],
            kg_facts=[],
            confidence=1.0,
            onboarding_active=True
        )

    # Classify intent
    intent, topics = classify_intent(query, session)
    log.info(f"[{cid[:8]}] Intent: {intent}, Topics: {[(t[0], f'{t[1]:.3f}') for t in topics[:2]]}")

    confidence = topics[0][1] if topics else 0.0

    # Load user profile
    profile = profile_manager.get_profile(user_id)

    # Handle setup profile
    if intent == "setup_profile":
        answer = onboarding_handler.start_onboarding(user_id)
        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[],
            chunks_used=[],
            kg_facts=[],
            confidence=1.0,
            onboarding_active=True
        )

    # Handle view profile
    if intent == "view_profile":
        if profile:
            answer = f"""**Your Profile:**

{profile.to_context_string()}

**Query Count:** {profile.query_count}
**Last Updated:** {profile.updated_at.strftime('%Y-%m-%d %H:%M')}

_To update your profile, use `/setup`_"""
        else:
            answer = "You don't have a profile yet. Use `/setup` to create one!"

        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[],
            chunks_used=[],
            kg_facts=[],
            confidence=1.0,
            profile_summary=profile.to_context_string() if profile else None
        )

    # Phase 3: Handle plan generation
    if intent == "generate_plan":
        if not profile or profile.entity_type == "Unknown":
            answer = """⚠️ **Profile Required**

To generate a compliance plan, I need to know more about your organization.

Please use `/setup` to create your profile first. I'll ask you a few questions about:
- Your entity type (NBFC, Bank, Fintech, etc.)
- Products you offer
- Business scale
- Digital lending operations

Once your profile is set up, I can create a customized compliance plan for you!"""
            return AskResponse(
                conversation_id=cid,
                user_id=user_id,
                answer=answer,
                intent=intent,
                matched_topics=[],
                chunks_used=[],
                kg_facts=[],
                confidence=1.0,
                profile_summary=None
            )

        # Generate compliance plan
        try:
            log.info(f"Generating compliance plan for user {user_id}")

            # Get relevant regulatory content for the profile
            # Use all applicable topics from compliance matrix
            requirements = compliance_matrix.get_applicable_requirements(profile)
            topic_keys = [req.topic_key for req in requirements[:5]]  # Top 5 topics

            # Retrieve relevant chunks
            all_chunks = []
            all_kg_facts = []
            for topic_key in topic_keys:
                chunks, kg_facts, _ = profile_aware_retrieve(
                    query=f"compliance requirements for {profile.entity_type}",
                    topic_key=topic_key,
                    profile=profile,
                    rag_search_func=rag_search,
                    kg_retrieve_func=hybrid_retrieve,
                    top_k=5,
                    rerank_alpha=PROFILE_RERANK_ALPHA
                )
                all_chunks.extend(chunks)
                all_kg_facts.extend(kg_facts)

            # Deduplicate chunks
            seen_ids = set()
            unique_chunks = []
            for chunk_id, text in all_chunks:
                if chunk_id not in seen_ids:
                    unique_chunks.append((chunk_id, text))
                    seen_ids.add(chunk_id)

            # Generate plan
            plan = plan_generator.generate_plan(
                profile=profile,
                chunks=unique_chunks[:20],  # Limit to 20 chunks
                kg_facts=all_kg_facts[:30],  # Limit to 30 KG facts
                plan_version=1
            )

            # Save plan
            plan_store.save(plan)

            # Format plan as markdown
            plan_dict = plan.to_dict()
            answer = PlanFormatter.to_markdown(plan_dict)

            log.info(f"Successfully generated plan {plan.plan_id} for user {user_id}")

            return AskResponse(
                conversation_id=cid,
                user_id=user_id,
                answer=answer,
                intent=intent,
                matched_topics=[{"topic": t, "label": t, "score": 1.0} for t in topic_keys[:3]],
                chunks_used=[{"id": c[0], "preview": c[1][:200]} for c in unique_chunks[:5]],
                kg_facts=all_kg_facts[:10],
                confidence=1.0,
                profile_summary=profile.to_context_string(),
                compliance_plan=plan_dict
            )

        except Exception as e:
            log.error(f"Plan generation failed: {e}")
            answer = f"⚠️ I encountered an error while generating your compliance plan: {str(e)}\n\nPlease try again or contact support."
            return AskResponse(
                conversation_id=cid,
                user_id=user_id,
                answer=answer,
                intent=intent,
                matched_topics=[],
                chunks_used=[],
                kg_facts=[],
                confidence=0.0,
                profile_summary=profile.to_context_string() if profile else None
            )

    # Handle view plan
    if intent == "view_plan":
        plans = plan_store.list_plans(user_id)
        if not plans:
            answer = """📋 **No Compliance Plans Found**

You haven't generated any compliance plans yet.

Use `/plan` or say "generate compliance plan" to create one!"""
        else:
            latest_plan = plans[0]
            plan_data = plan_store.load(user_id, version=latest_plan["version"])
            if plan_data:
                answer = PlanFormatter.to_markdown(plan_data)
            else:
                answer = "⚠️ Could not load plan. Please try generating a new one."

        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[],
            chunks_used=[],
            kg_facts=[],
            confidence=1.0,
            profile_summary=profile.to_context_string() if profile else None
        )

    # Handle plan refinement
    if intent == "refine_plan":
        # Extract feedback from query
        feedback = query.replace("/refine-plan", "").replace("refine plan", "").replace("update plan", "").strip()

        if not feedback:
            answer = """📝 **Plan Refinement**

To refine your compliance plan, please provide specific feedback.

Examples:
- "Focus more on digital lending requirements"
- "Add more details on KYC compliance"
- "Prioritize gold loan regulations"
- "We don't offer microfinance, remove that section"

Usage: `/refine-plan <your feedback>`"""
        else:
            # Load latest plan
            plans = plan_store.list_plans(user_id)
            if not plans:
                answer = "⚠️ No existing plan found. Please generate a plan first using `/plan`"
            else:
                try:
                    # Load original plan
                    latest_plan_data = plan_store.load(user_id, version=plans[0]["version"])

                    # Create a CompliancePlan object from dict (simplified)
                    from plan_generator import CompliancePlan, PriorityArea, ActionItem, RiskArea
                    original_plan = CompliancePlan(
                        plan_id=latest_plan_data["plan_id"],
                        user_id=latest_plan_data["user_id"],
                        generated_at=datetime.fromisoformat(latest_plan_data["generated_at"]),
                        version=latest_plan_data["version"],
                        entity_type=latest_plan_data["entity_type"],
                        products=latest_plan_data["products"],
                        summary=latest_plan_data["summary"],
                        priority_areas=[PriorityArea(**pa) for pa in latest_plan_data["priority_areas"]],
                        timeline_based_actions={
                            k: [ActionItem(**ai) for ai in v]
                            for k, v in latest_plan_data["timeline_based_actions"].items()
                        },
                        risk_areas=[RiskArea(**ra) for ra in latest_plan_data["risk_areas"]],
                        implementation_checklist=latest_plan_data["implementation_checklist"],
                        requirements_count=latest_plan_data["requirements_count"],
                        applicable_topics=latest_plan_data["applicable_topics"]
                    )

                    # Refine plan
                    refined_plan = plan_generator.refine_plan(
                        original_plan=original_plan,
                        user_feedback=feedback,
                        profile=profile
                    )

                    # Save refined plan
                    plan_store.save(refined_plan)

                    # Format as markdown
                    answer = PlanFormatter.to_markdown(refined_plan.to_dict())

                    log.info(f"Successfully refined plan to version {refined_plan.version} for user {user_id}")

                except Exception as e:
                    log.error(f"Plan refinement failed: {e}")
                    answer = f"⚠️ Plan refinement failed: {str(e)}\n\nPlease try again or generate a new plan."

        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[],
            chunks_used=[],
            kg_facts=[],
            confidence=1.0,
            profile_summary=profile.to_context_string() if profile else None
        )

    # Handle greeting
    if intent == "greeting":
        answer = greeting_response(query, profile)
        update_session(cid, query, answer, [])
        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[],
            chunks_used=[],
            kg_facts=[],
            confidence=1.0,
            profile_summary=profile.to_context_string() if profile else None
        )

    # Handle out of scope
    if intent == "out_of_scope":
        answer = out_of_scope_response(query, topics)
        update_session(cid, query, answer, [])
        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[{"topic": t[0], "label": topic_matcher.get_topic_label(t[0]), "score": round(t[1], 3)} for t in topics[:3]],
            chunks_used=[],
            kg_facts=[],
            confidence=confidence,
            profile_summary=profile.to_context_string() if profile else None
        )

    # Handle follow-up
    if intent == "followup":
        original_query = query
        query = rewrite_followup(query, session)
        log.info(f"[{cid[:8]}] Rewritten: '{original_query}' → '{query}'")
        topics = topic_matcher.match_multiple(query, top_k=3)
        confidence = topics[0][1] if topics else 0.0

    # Get primary topic
    primary_topic = topics[0][0] if topics else None

    # NEW: Profile-aware retrieval
    try:
        chunks, kg_facts, debug_info = profile_aware_retrieve(
            query=query,
            topic_key=primary_topic,
            profile=profile,
            rag_search_func=rag_search,
            kg_retrieve_func=hybrid_retrieve,
            top_k=10,
            rerank_alpha=PROFILE_RERANK_ALPHA,
            filter_entity=False  # Set to True for strict filtering
        )

        log.info(f"[{cid[:8]}] Retrieved {len(chunks)} chunks (reranked: {debug_info.get('rerank_applied')})")
    except Exception as e:
        log.error(f"[{cid[:8]}] Retrieval failed: {e}")
        chunks, kg_facts = [], []

    # Handle no results
    if not chunks and not kg_facts:
        answer = f"""I couldn't find specific information about this in the RBI documents.

**Detected topic:** {topic_matcher.get_topic_label(primary_topic) if primary_topic else 'Unknown'}

Could you try:
• Rephrasing your question more specifically
• Asking about a related RBI regulation
• Checking if your question falls under a different topic"""

        update_session(cid, query, answer, [t[0] for t in topics])
        return AskResponse(
            conversation_id=cid,
            user_id=user_id,
            answer=answer,
            intent=intent,
            matched_topics=[{"topic": t[0], "label": topic_matcher.get_topic_label(t[0]), "score": round(t[1], 3)} for t in topics[:3]],
            chunks_used=[],
            kg_facts=[],
            confidence=confidence,
            profile_summary=profile.to_context_string() if profile else None
        )

    # Generate answer (with profile context)
    answer = generate_answer(query, chunks, kg_facts, topics, profile)

    # Update profile from interaction
    if profile:
        topic_keys = [t[0] for t in topics]
        profile_manager.update_from_interaction(user_id, query, topic_keys)
    else:
        # Create profile automatically from query
        topic_keys = [t[0] for t in topics]
        profile = profile_manager.update_from_interaction(user_id, query, topic_keys)

    # Update session
    topic_keys = [t[0] for t in topics]
    update_session(cid, query, answer, topic_keys)

    # Analyze readability of response
    readability_score = analyze_response_readability(answer)

    return AskResponse(
        conversation_id=cid,
        user_id=user_id,
        answer=answer,
        intent=intent,
        matched_topics=[
            {
                "topic": t[0],
                "label": topic_matcher.get_topic_label(t[0]),
                "score": round(t[1], 3)
            }
            for t in topics[:3]
        ],
        chunks_used=[{"id": c[0], "preview": c[1][:300]} for c in chunks[:5]],
        kg_facts=kg_facts[:10],
        confidence=round(confidence, 3),
        profile_summary=profile.to_context_string() if profile else None,
        readability=readability_score
    )

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "8.0.0",
        "model": LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "topics_loaded": len(topic_matcher.get_all_topics()),
        "features": {
            "dynamic_topic_matching": True,
            "hardcoded_rules": False,
            "multi_topic_support": True,
            "session_management": True,
            "follow_up_detection": True,
            "confidence_scoring": True,
            "user_profiling": True,
            "profile_aware_retrieval": True,
            "onboarding_flow": True,
            "compliance_plan_generation": True,  # Phase 3
            "plan_versioning": True,  # Phase 3
            "plan_refinement": True,  # Phase 3
            "multi_format_output": True,  # Phase 3
        }
    }

@app.get("/topics")
def list_topics():
    topics = []
    for topic_key in sorted(topic_matcher.get_all_topics()):
        topics.append({
            "key": topic_key,
            "label": topic_matcher.get_topic_label(topic_key)
        })
    return {
        "count": len(topics),
        "topics": topics
    }

@app.post("/debug/match")
def debug_topic_match(request: Dict[str, str]):
    query = request.get("query", "")
    if not query:
        raise HTTPException(400, "Query required")

    matches = topic_matcher.match_multiple(query, top_k=10)

    return {
        "query": query,
        "top_matches": [
            {
                "rank": i + 1,
                "topic": m[0],
                "label": topic_matcher.get_topic_label(m[0]),
                "score": round(m[1], 4)
            }
            for i, m in enumerate(matches)
        ]
    }

@app.delete("/session/{conversation_id}")
def delete_session(conversation_id: str):
    clear_session(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}

# NEW: Profile endpoints
@app.get("/profile/{user_id}")
def get_user_profile(user_id: str):
    """Get user profile"""
    profile = profile_manager.get_profile(user_id)
    if not profile:
        raise HTTPException(404, f"Profile not found for user {user_id}")
    return profile

@app.post("/profile/{user_id}/setup")
def setup_profile(user_id: str):
    """Start profile setup"""
    answer = onboarding_handler.start_onboarding(user_id)
    return {"user_id": user_id, "message": answer, "onboarding_active": True}

@app.delete("/profile/{user_id}")
def delete_user_profile(user_id: str):
    """Delete user profile"""
    success = profile_manager.delete_profile(user_id)
    if not success:
        raise HTTPException(404, f"Profile not found for user {user_id}")
    return {"status": "deleted", "user_id": user_id}

# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    import uvicorn
    log.info("🚀 Starting RBI Chatbot V7 (with User Profiling)...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
