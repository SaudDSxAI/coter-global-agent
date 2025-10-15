import streamlit as st
from pathlib import Path
import base64
from bot_backend import run_pipeline

# --- Page setup ---
st.set_page_config(page_title="CoterGlobal AI", layout="wide")

# --- Custom CSS (Elegant Slate Theme) ---
st.markdown("""
    <style>
        /* Global Theme Fix — full blue background */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #009aee !important;
            background: linear-gradient(145deg, #009aee 0%, #00b3ff 100%) !important;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
            height: 100%;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Remove white/black top & bottom spaces */
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
            background: transparent !important;
        }

        /* Adjust main content container */
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            background: transparent !important;
        }

        /* Profile Section */
        .profile-container {
            display: flex;
            justify-content: center;
            margin: 25px 0 10px 0;
        }
        .profile-pic {
            width: 140px;
            height: 140px;
            object-fit: cover;
            object-position: center 15%;
            clip-path: circle(45% at 50% 50%);
            border: 3px solid #0A66C2;
            box-shadow: 0 4px 10px rgba(10,102,194,0.2);
        }

        /* Titles */
        .main-title {
            text-align: center;
            font-size: 1.8em;
            font-weight: 600;
            color: #000b31;
            margin-top: 12px;
            letter-spacing: -0.3px;
        }

        .sub-title {
            text-align: center;
            font-size: 1em;
            color: #000000;
            margin-bottom: 35px;
            font-weight: 400;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            background-color: #c2d0ff;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.04);
            color: #1e293b !important;
        }

        /* Assistant message */
        div[data-testid="stChatMessage"]:has([data-testid="assistant"]) {
            border-left: 4px solid #004c99;
        }

        /* User message */
        div[data-testid="stChatMessage"]:has([data-testid="user"]) {
            background-color: #ffffff;
            border-left: 4px solid #009aee;
        }

        /* Message Text */
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div {
            color: #1e293b !important;
            font-size: 0.98em;
            line-height: 1.45;
        }

        /* Chat Input */
        .stChatInput textarea {
            background-color: #ffffff !important;
            color: #1e293b !important;
            border-radius: 10px !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 3px 6px rgba(10,102,194,0.08);
            font-family: 'Inter', sans-serif !important;
            font-size: 1.05em !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb {
            background: #b3caec;
            border-radius: 10px;
        }

        /* Hover Effect */
        div[data-testid="stChatMessage"]:hover {
            box-shadow: 0 6px 16px rgba(10,102,194,0.08);
            transform: translateY(-1px);
            transition: 0.25s ease-in-out;
        }

        /* Fade-in Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .assistant-response {
            animation: fadeIn 0.6s ease-out;
        }

        /* Hide Default Avatars */
        [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)




# --- Initialize assistant ---
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = run_pipeline(init_only=True)
    st.session_state["messages"] = []

qa_chain = st.session_state["qa_chain"]

# --- Display Profile Image ---
def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

image_path = Path(__file__).parent / "coter_logo.jpeg"
if image_path.exists():
    img_base64 = image_to_base64(image_path)
    st.markdown(f"""
        <div class="profile-container">
            <img src="data:image/jpeg;base64,{img_base64}" class="profile-pic">
        </div>
    """, unsafe_allow_html=True)

# --- Titles ---
st.markdown("<div class='main-title'>Hello dear!</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>I am a chatbot, ask questions and share your data here.</div>", unsafe_allow_html=True)

# --- Chat Container ---
chat_container = st.container()

# --- Render Messages ---
with chat_container:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(f"<div class='assistant-response'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

# --- Chat input ---
if prompt := st.chat_input("Type your question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("💭 *Thinking...*")

        try:
            result = qa_chain.invoke({"query": prompt})
            response = result["result"] if isinstance(result, dict) else result
        except Exception as e:
            response = f"⚠️ Error: {e}"

        message_placeholder.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})