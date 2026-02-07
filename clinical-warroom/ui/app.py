"""
Clinical War Room - Main Streamlit App

ChatGPT-style interface for doctor queries with multi-agent deliberation.

Workflow:
1. Add Patient - Upload PDF or enter text
2. Query Patient - Chat with visible agent discussion

Usage:
    streamlit run ui/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Page config
st.set_page_config(
    page_title="Clinical War Room",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
    }
    
    .chat-message {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
    }
    
    .risk-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-low { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e; }
    .risk-moderate { background: #eab30822; color: #eab308; border: 1px solid #eab308; }
    .risk-high { background: #ef444422; color: #ef4444; border: 1px solid #ef4444; }
    .risk-critical { background: #dc262622; color: #dc2626; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the War Room header."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🏥 Clinical War Room")
        st.caption("Multi-Agent Decision Support System")
    with col2:
        st.success("● LIVE")


def render_upload_patient_tab():
    """Render the upload/add patient interface."""
    st.markdown("## 📤 Add New Patient")
    st.caption("Upload a document or enter patient details as text")
    
    from rag.patient_store import get_patient_store
    store = get_patient_store()
    
    tab1, tab2 = st.tabs(["📄 Upload File", "📝 Enter Text"])
    
    with tab1:
        st.markdown("**Supported formats:** PDF, DOCX, TXT, PNG, JPG")
        
        uploaded_file = st.file_uploader(
            "Upload Patient Document",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            help="Upload a document containing patient medical records"
        )
        
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if st.button("📥 Process File", type="primary"):
                with st.spinner(f"Processing {file_ext.upper()} file..."):
                    try:
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        patient_id = store.add_patient_from_file(tmp_path)
                        os.unlink(tmp_path)
                        
                        st.success(f"✅ Patient added: **{patient_id}**")
                        st.balloons()
                        
                        patient = store.get_patient(patient_id)
                        if patient:
                            with st.expander("View Extracted Text"):
                                st.text(patient.raw_text[:2000] + "..." if len(patient.raw_text) > 2000 else patient.raw_text)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        with st.form("add_patient_form"):
            patient_name = st.text_input("Patient Name", placeholder="John Doe")
            
            col1, col2 = st.columns(2)
            with col1:
                patient_age = st.number_input("Age", min_value=1, max_value=120, value=65)
            with col2:
                patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            medical_history = st.text_area(
                "Patient Details / Medical History",
                placeholder="Enter all patient information here...",
                height=200
            )
            
            st.markdown("**Optional: Gait Measurements**")
            col1, col2, col3 = st.columns(3)
            with col1:
                stride_length = st.number_input("Stride Length (m)", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
            with col2:
                gait_speed = st.number_input("Gait Speed (m/s)", min_value=0.0, max_value=3.0, value=0.0, step=0.01)
            with col3:
                symmetry = st.number_input("Symmetry Index", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            
            submitted = st.form_submit_button("➕ Add Patient", type="primary")
            
            if submitted and medical_history:
                gait_data = {}
                if stride_length > 0:
                    gait_data["stride_length"] = stride_length
                if gait_speed > 0:
                    gait_data["gait_speed"] = gait_speed
                if symmetry > 0:
                    gait_data["symmetry_index"] = symmetry
                
                full_text = f"Name: {patient_name}\nAge: {patient_age}\nGender: {patient_gender}\n\n{medical_history}"
                
                patient_id = store.add_patient_from_text(
                    text=full_text,
                    name=patient_name,
                    age=patient_age,
                    gender=patient_gender,
                    gait_data=gait_data if gait_data else None,
                )
                
                st.success(f"✅ Patient added: **{patient_id}**")
                st.balloons()
    
    # Show existing patients
    st.markdown("---")
    st.markdown("## 📋 Stored Patients")
    
    patients = store.list_patients()
    if patients:
        for p in patients:
            with st.expander(f"🧑‍⚕️ {p['name']} ({p['patient_id']})"):
                st.markdown(f"**Age:** {p['age']} | **Gender:** {p['gender']}")
    else:
        st.info("No patients in database. Add a patient above.")


def render_chat_interface():
    """Render the ChatGPT-style query interface."""
    st.markdown("## 💬 Query Patient")
    st.caption("Ask questions about a patient - see the multi-agent deliberation")
    
    from rag.patient_store import get_patient_store
    from core.chat_engine import get_chat_engine
    
    store = get_patient_store()
    engine = get_chat_engine()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None
    
    # Patient selector
    patients = store.list_patients()
    if not patients:
        st.warning("⚠️ No patients in database. Add a patient first.")
        return
    
    patient_options = {p['patient_id']: f"{p['name']} ({p['patient_id']})" for p in patients}
    
    selected_id = st.selectbox(
        "Select Patient:",
        options=list(patient_options.keys()),
        format_func=lambda x: patient_options[x]
    )
    
    if selected_id:
        patient = store.get_patient(selected_id)
        st.session_state.selected_patient = patient
        
        # Show patient summary
        with st.expander("📋 Patient Context", expanded=False):
            st.markdown(f"**Name:** {patient.name}")
            st.markdown(f"**Age:** {patient.age} | **Gender:** {patient.gender}")
            st.markdown(f"**Medical History:**")
            st.text(patient.raw_text[:500] + "..." if len(patient.raw_text) > 500 else patient.raw_text)
    
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👨‍⚕️ Doctor:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            response = msg.get("response")
            if response:
                render_chat_response(response)
    
    # Chat input
    st.markdown("---")
    
    # User input
    user_query = st.chat_input("Ask about the patient...")
    
    if user_query and patient:
        process_query(user_query, patient, engine)


def process_query(query, patient, engine):
    """Process a user query and display results."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get response from LLM + RL
    with st.spinner("🔄 Agents analyzing..."):
        response = engine.query(
            question=query,
            patient_context=patient.raw_text,
            patient_id=patient.patient_id,
            patient_name=patient.name,
            gait_data=patient.gait_data,
        )
    
    # Add assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "response": response
    })
    
    st.rerun()


def render_chat_response(response):
    """Render a chat response with agent deliberation."""
    risk_class = f"risk-{response.risk_level}"
    
    st.markdown(f"""
    <div class="chat-message assistant-message">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <strong>🤖 Clinical War Room</strong>
            <span class="risk-badge {risk_class}">{response.risk_level.upper()} RISK</span>
        </div>
        <div style="margin-bottom: 12px;">{response.answer}</div>
        <div style="font-size: 0.9em; color: #9ca3af;">
            Confidence: {response.confidence:.0%} | Recommendation: <strong>{response.recommendation}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk factors
    if response.risk_factors:
        st.markdown("**Risk Factors:**")
        for rf in response.risk_factors:
            st.markdown(f"- {rf}")
    
    # Agent deliberation - expandable
    with st.expander("🔍 View Agent Deliberation", expanded=False):
        render_agent_deliberation(response.agent_messages)
    
    # RL explanation
    if response.rl_explanation:
        with st.expander("🧠 RL Decision Explanation", expanded=False):
            st.info(response.rl_explanation)
            if response.rl_was_overridden:
                st.warning("⚠️ Safety override was applied")


def render_agent_deliberation(messages):
    """Render the agent deliberation messages."""
    agent_icons = {
        "Diagnostic Agent": "🔬",
        "Risk Agent": "⚠️",
        "Evidence Agent": "📚",
        "Ethics Agent": "⚖️",
        "Data Quality Agent": "📊",
        "RL Coordinator": "🧠",
    }
    
    for msg in messages:
        icon = agent_icons.get(msg.agent_name, "🤖")
        
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"### {icon}")
            with col2:
                st.markdown(f"**{msg.agent_name}**")
                st.markdown(msg.message)
                if msg.confidence > 0:
                    st.caption(f"Confidence: {msg.confidence:.0%}")
            st.markdown("---")


def main():
    """Main app entry point."""
    render_header()
    
    # Mode selector
    st.sidebar.markdown("## 🎮 Mode")
    mode = st.sidebar.radio(
        "Select:",
        options=["chat", "upload"],
        format_func=lambda x: {
            "chat": "💬 Query Patient",
            "upload": "📤 Add Patient", 
        }[x],
    )
    
    st.sidebar.markdown("---")
    
    # Show patient count
    try:
        from rag.patient_store import get_patient_store
        store = get_patient_store()
        st.sidebar.metric("Patients in DB", store.count())
    except:
        pass
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ System Info")
    st.sidebar.caption("• LLM: Groq API")
    st.sidebar.caption("• Storage: ChromaDB")
    st.sidebar.caption("• RL: Trained Q-Learning")
    
    if mode == "chat":
        render_chat_interface()
    else:
        render_upload_patient_tab()
    
    # Footer
    st.markdown("---")
    st.caption("Clinical War Room v1.0 | Multi-Agent Decision Support")


if __name__ == "__main__":
    main()
