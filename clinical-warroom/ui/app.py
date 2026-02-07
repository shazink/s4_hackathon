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
    footer {visibility: hidden;}
    
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Main Background - Light Blue */
    .stApp {
        background-color: #e0f2fe; /* Light Blue */
    }
    
    /* Headers - Black & Orange Accent */
    h1 {
        color: #000000 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 0px #fdba74; /* Orange Shadow */
    }
    
    h2, h3, h4 {
        color: #ea580c !important; /* Burnt Orange */
        font-weight: 700 !important;
    }
    
    /* Text - Black */
    p, li, span, div {
        color: #000000;
        font-weight: 500;
    }
    
    /* Sidebar styling - White with Orange/Red borders */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 3px solid #f97316; /* Orange */
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 18px 24px;
        border-radius: 16px;
        margin: 16px 0;
        box-shadow: 4px 4px 0px rgba(0,0,0,0.1); /* Hard shadow */
        color: #000000;
        border: 2px solid #000000;
    }
    
    .chat-message:hover {
        transform: translate(-1px, -1px);
        box-shadow: 6px 6px 0px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: #fdba74; /* Orange */
        color: #000000;
        margin-left: 10%;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 2px solid #000000;
        border-left: 8px solid #ef4444; /* Red */
        color: #000000;
        margin-right: 10%;
    }
    
    /* Risk badges - Pop style */
    .risk-badge {
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8em;
        font-weight: 800;
        text-transform: uppercase;
        border: 2px solid #000000;
    }
    
    .risk-low { 
        background: #6ee7b7; /* Green */
        color: #000000; 
    }
    .risk-moderate { 
        background: #fdba74; /* Orange */
        color: #000000; 
    }
    .risk-high { 
        background: #fca5a5; /* Red */
        color: #000000; 
    }
    .risk-critical { 
        background: #ef4444; 
        color: #ffffff; 
        animation: pulse 1s infinite alternate;
    }
    
    @keyframes pulse {
        from { transform: scale(1); }
        to { transform: scale(1.05); }
    }
    
    /* Button styling - Red/Orange Gradient */
    .stButton>button {
        background: linear-gradient(45deg, #f97316, #ef4444);
        color: #ffffff !important;
        border: 2px solid #000000;
        border-radius: 8px;
        font-weight: 800;
        padding: 0.6rem 1.6rem;
        box-shadow: 3px 3px 0px #000000;
        transition: all 0.1s ease;
    }
    
    .stButton>button:hover {
        transform: translate(-1px, -1px);
        box-shadow: 5px 5px 0px #000000;
    }
    
    .stButton>button:active {
        transform: translate(2px, 2px);
        box-shadow: 1px 1px 0px #000000;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        color: #000000;
        background-color: #ffffff;
        border: 2px solid #000000;
        border-radius: 8px;
        box-shadow: 3px 3px 0px #bae6fd; /* Light Blue shadow */
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the professional medical header."""
    st.markdown("""
    <div style="padding: 10px 0;">
        <h1 style="margin: 0; color: #1e40af; font-size: 2.5em;">🏥 Clinical Decision Support Platform</h1>
        <p style="margin: 5px 0 0 0; color: #64748b; font-size: 1.1em; font-weight: 500;">AI-Powered Multi-Agent Clinical Analysis System</p>
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
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_patient" not in st.session_state:
        st.session_state.selected_patient = None
    
    with st.sidebar:
        st.markdown("---")

    # Patient Selector (Main Page)
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
    
    # --- MAIN CHAT AREA ---
    if not st.session_state.selected_patient:
        st.info("👆 Please select a patient above to begin analysis.")
        return
        
    patient = st.session_state.selected_patient
    
    # Main chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👨‍⚕️ Doctor</strong><br>
                    {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                render_chat_response(msg["response"])
    
    # Chat input
    if user_query := st.chat_input(f"Assess {patient.name}..."):
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
                        <div style="color: #334155; font-size: 1em; line-height: 1.5;">
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
        <div style="color: #64748b; font-size: 0.95em; font-weight: 500;">
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
    # Professional mode selector
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
    

    
    if mode == "chat":
        render_chat_interface()
    else:
        render_upload_patient_tab()
    
    # Professional footer
    st.markdown("---")
    st.caption("Clinical Decision Support Platform v2.0 | AI-Powered Multi-Agent Analysis")


if __name__ == "__main__":
    main()
