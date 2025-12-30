import streamlit as st
import requests
import uuid

# =====================
# CONFIG
# =====================
API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="RBI Regulatory Intelligence Platform",
    page_icon="🏦",
    layout="wide"
)

# =====================
# SESSION STATE
# =====================
if "cid" not in st.session_state:
    st.session_state.cid = str(uuid.uuid4())

if "user_id" not in st.session_state:
    st.session_state.user_id = st.session_state.cid

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================
# SIDEBAR (MANAGER VIEW)
# =====================
with st.sidebar:
    st.title("📊 Demo Dashboard")

    st.markdown("### Enabled Phases")
    st.success("Phase 1 – Intelligence & RAG")
    st.success("Phase 2 – User Profiling")
    st.success("Phase 3 – Compliance Planning")
    st.success("NEW – Readability Analysis")

    st.markdown("---")
    st.markdown("### Supported Commands")
    st.code("""
/setup
/profile
/plan
/view-plan
/refine-plan <feedback>
""")

    if st.button("🔄 Reset Demo"):
        st.session_state.cid = str(uuid.uuid4())
        st.session_state.user_id = st.session_state.cid
        st.session_state.messages = []
        st.rerun()

# =====================
# HEADER
# =====================
st.title("🏦 RBI Regulatory Intelligence Platform")
st.caption(
    "Retrieval → Personalization → Compliance Action Plans (Enterprise Demo)"
)

# =====================
# CHAT HISTORY
# =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================
# USER INPUT
# =====================
query = st.chat_input("Ask about RBI regulations or type a command...")

if query:
    # --- User message ---
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    payload = {
        "question": query,
        "conversation_id": st.session_state.cid,
        "user_id": st.session_state.user_id
    }

    try:
        res = requests.post(API_URL, json=payload, timeout=120)
        data = res.json()
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Backend error: {e}")
        st.stop()

    # --- Assistant message ---
    with st.chat_message("assistant"):
        st.markdown(data.get("answer", ""))

        # =====================
        # MANAGER / DEBUG VIEW
        # =====================
        with st.expander("🧠 Phase-wise Breakdown (Manager View)", expanded=False):

            col1, col2 = st.columns(2)

            # -------- Phase 1 --------
            with col1:
                st.markdown("### Phase 1 – Intelligence")
                st.write("**Intent:**", data.get("intent"))
                st.progress(min(data.get("confidence", 0.0), 1.0))

                if data.get("matched_topics"):
                    st.markdown("**Detected Topics:**")
                    for t in data["matched_topics"]:
                        st.write(
                            f"- {t.get('label', t.get('topic'))} ({t.get('score')})"
                        )

            # -------- Phase 2 --------
            with col2:
                st.markdown("### Phase 2 – User Profile")
                if data.get("profile_summary"):
                    st.success("Profile Active")
                    st.info(data["profile_summary"])
                else:
                    st.warning("No profile detected")

                if data.get("onboarding_active"):
                    st.warning("Onboarding in progress")

            # -------- Phase 1: Retrieval --------
            if data.get("chunks_used"):
                st.markdown("### 📄 Retrieved RBI Chunks")
                for c in data["chunks_used"]:
                    st.markdown(f"**{c.get('id')}**")
                    st.caption(c.get("preview", "") + "...")

            if data.get("kg_facts"):
                st.markdown("### 🧩 Knowledge Graph Facts")
                st.json(data["kg_facts"])

            # -------- Phase 3: Compliance Plan --------
            plan = data.get("compliance_plan")
            if plan:
                st.markdown("### 📋 Phase 3 – Compliance Plan")
                st.success("Compliance Plan Generated")

                st.markdown(f"**Entity Type:** {plan.get('entity_type')}")
                st.markdown(f"**Requirements Covered:** {plan.get('requirements_count')}")

                st.markdown("#### Applicable Regulatory Areas")
                for t in plan.get("applicable_topics", []):
                    st.write(f"- {t}")

                st.markdown("#### Priority Areas")
                for p in plan.get("priority_areas", []):
                    label = (
                        p.get("title")
                        or p.get("area")
                        or p.get("name")
                        or "Priority Area"
                    )
                    risk = p.get("risk_level", "Unknown")
                    desc = p.get("description", "")

                    st.markdown(f"**🔹 {label}**")
                    st.write("Risk Level:", risk)
                    if desc:
                        st.caption(desc)

                st.markdown("#### Timeline-based Actions")
                for phase, actions in plan.get("timeline_based_actions", {}).items():
                    st.markdown(f"**{phase}**")
                    for a in actions:
                        action_text = a.get("action") or a.get("task") or "Action item"
                        st.write("•", action_text)

            # -------- NEW: Readability --------
            readability = data.get("readability")
            if readability:
                st.markdown("### 📖 Response Readability (NEW)")
                st.write("**Readability Level:**", readability.get("readability_level"))
                st.write("**Grade Level:**", readability.get("grade_level"))
                st.write("**Flesch Reading Ease:**", readability.get("flesch_reading_ease"))
                st.write("**Flesch-Kincaid Grade:**", readability.get("flesch_kincaid_grade"))
                st.write(
                    "Words:",
                    readability.get("total_words"),
                    "| Sentences:",
                    readability.get("total_sentences"),
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": data.get("answer", "")}
    )
