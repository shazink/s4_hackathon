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
    page_title="Clinical Decision Support Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - Professional Medical Theme
st.markdown("""
<style>
    /* Hide Streamlit default header, menu, and footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Remove top padding to make room for custom header */
    .block-container {
        padding-top: 1rem;
    }
    
    .stApp {
        background-color: #0a1628;
    }
    
    /* Professional header styling */
    h1 {
        color: #60a5fa !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    h2, h3 {
        color: #93c5fd !important;
        font-weight: 500 !important;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        margin-left: 15%;
        border-left: 4px solid #3b82f6;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 4px solid #60a5fa;
        color: #e2e8f0;
    }
    
    /* Risk badges - medical colors */
    .risk-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    
    .risk-low { 
        background: #064e3b; 
        color: #6ee7b7; 
        border: 2px solid #10b981; 
    }
    .risk-moderate { 
        background: #713f12; 
        color: #fcd34d; 
        border: 2px solid #f59e0b; 
    }
    .risk-high { 
        background: #7f1d1d; 
        color: #fca5a5; 
        border: 2px solid #ef4444; 
    }
    .risk-critical { 
        background: #450a0a; 
        color: #fca5a5; 
        border: 2px solid #dc2626;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the professional medical header."""
    col1, col2, col3 = st.columns([6, 2, 1])
    with col1:
        st.markdown("""
        <div style="padding: 10px 0;">
            <h1 style="margin: 0; color: #60a5fa; font-size: 2.2em;">🏥 Clinical Decision Support Platform</h1>
            <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 0.95em;">AI-Powered Multi-Agent Clinical Analysis System</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="padding: 20px 0; text-align: right;">
            <span style="color: #10b981; font-weight: 600;">● SYSTEM ONLINE</span>
        </div>
        """, unsafe_allow_html=True)


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
    </div>
    """, unsafe_allow_html=True)
    
    # Multi-Agent Debate Section
    if response.debate_rounds:
        render_debate_section(response.debate_rounds, response.treatment_urgency)
    
    
    # Agent deliberation - expandable
    with st.expander("🔍 View Agent Deliberation", expanded=False):
        render_agent_deliberation(response.agent_messages)
    
    

def render_debate_section(debate_rounds, treatment_urgency):
    """Render the multi-agent debate in 3 columns."""
    st.markdown("### 💬 Multi-Agent Debate: Treatment Urgency")
    st.markdown("---")
    
    # Agent colors
    agent_colors = {
        "proponent": "#10b981",  # green
        "skeptic": "#ef4444",    # red
        "mediator": "#f59e0b"    # yellow/orange
    }
    
    agent_names = {
        "proponent": "🟢 Proponent",
        "skeptic": "🔴 Skeptic",
        "mediator": "🟡 Mediator"
    }
    
    # Display each debate round
    for round_data in debate_rounds:
        st.markdown(f"#### Round {round_data['round']}: {round_data['title']}")
        
        # Create 3 columns for the 3 agents
        col1, col2, col3 = st.columns(3)
        
        cols = [col1, col2, col3]
        agent_keys = ["proponent", "skeptic", "mediator"]
        
        for i, (col, agent_key) in enumerate(zip(cols, agent_keys)):
            with col:
                agent_message = round_data["agents"].get(agent_key, "")
                if agent_message:
                    color = agent_colors[agent_key]
                    name = agent_names[agent_key]
                    
                    # Speech bubble style
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                        border-left: 4px solid {color};
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 12px;
                        min-height: 100px;
                    ">
                        <div style="font-weight: bold; color: {color}; margin-bottom: 8px;">
                            {name}
                        </div>
                        <div style="color: #e5e7eb; font-size: 0.95em;">
                            {agent_message}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Final Consensus Card
    st.markdown("---")
    urgency_colors = {
        "IMMEDIATE": ("#ef4444", "🚨 IMMEDIATE TREATMENT NEEDED"),
        "ESCALATE": ("#f59e0b", "⚠️ ESCALATE TO SPECIALIST"),
        "MONITOR": ("#10b981", "📋 CONTINUE MONITORING")
    }
    
    color, label = urgency_colors.get(treatment_urgency, ("#6b7280", "⚪ NO DECISION"))
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}33 0%, {color}22 100%);
        border: 2px solid {color};
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
    ">
        <div style="font-size: 1.5em; font-weight: bold; color: {color}; margin-bottom: 8px;">
            {label}
        </div>
        <div style="color: #9ca3af; font-size: 0.9em;">
            Consensus reached through multi-agent deliberation
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")


def render_agent_deliberation(messages):
    """Render the agent deliberation messages."""
    agent_icons = {
        "Diagnostic Agent": "🔬",
        "Risk Agent": "⚠️",
        "Evidence Agent": "📚",
        "Ethics Agent": "⚖️",
        "Data Quality Agent": "📊",
        "Decision System": "🤖",
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
            st.markdown("---")


def main():
    """Main app entry point."""
    render_header()
    
    # Professional mode selector
    st.sidebar.markdown("## ⚙️ System Controls")
    mode = st.sidebar.radio(
        "Select Mode:",
        options=["chat", "upload"],
        format_func=lambda x: {
            "chat": "💬 Clinical Query",
            "upload": "📥 Patient Registration", 
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
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.caption("• AI Engine: Groq API (Llama 3.3)")
    st.sidebar.caption("• Vector DB: ChromaDB")
    st.sidebar.caption("• Agents: 5 Analysis + 3 Debate")
    st.sidebar.caption("• Decision: Confidence-Based")
    
    if mode == "chat":
        render_chat_interface()
    else:
        render_upload_patient_tab()
    
    # Professional footer
    st.markdown("---")
    st.caption("Clinical Decision Support Platform v2.0 | AI-Powered Multi-Agent Analysis")


if __name__ == "__main__":
    main()
