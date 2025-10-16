import streamlit as st
from pathlib import Path
import base64
from bot_backend import run_pipeline

# --- Page setup ---
st.set_page_config(page_title="CoterGlobal AI", layout="wide")

# --- Custom CSS (Elegant Slate Theme) ---
st.markdown("""
    <style>
        /* === FULL BACKGROUND THEME === */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #0077b6 !important;
            background: linear-gradient(135deg, #0077b6 0%, #0096c7 50%, #00b4d8 100%) !important;
            color: white !important;
            height: 100% !important;
            min-height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* === REMOVE HEADER / FOOTER / TOOLBAR === */
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        footer,
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        /* === REMOVE INTERNAL BOTTOM PADDING === */
        section.main, div.block-container {
            padding-bottom: 0 !important;
            margin-bottom: 0 !important;
            background: transparent !important;
        }

        /* === FULL APP CONTAINER FIX === */
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }

        /* === BODY FIX FOR DARK PADDING === */
        [data-testid="stVerticalBlock"] {
            background: transparent !important;
        }

        /* === CENTER LOGO + TITLES === */
        .center-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            margin-bottom: 40px;
        }

        .profile-pic {
            width: 140px;
            height: 140px;
            object-fit: cover;
            object-position: center 15%;
            clip-path: circle(45% at 50% 50%);
            border: 3px solid #0A66C2;
            box-shadow: 0 4px 10px rgba(10,102,194,0.2);
            margin-bottom: 10px;
        }

        .main-title {
            font-size: 2.4em;
            font-weight: 700;
            color: #ffffff;
            margin-top: 10px;
        }

        .sub-title {
            font-size: 1.1em;
            color: #e0f7ff;
            margin-bottom: 25px;
        }

        /* === CHAT UI === */
        .chat-bubble {
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 20px;
            margin: 10px auto;
            font-size: 1.05em;
            line-height: 1.6;
        }

        .user-bubble {
            background-color: rgba(255, 255, 255, 0.9);
            color: #0077b6;
            border-bottom-right-radius: 5px;
        }

        .ai-bubble {
            background-color: rgba(10, 102, 194, 0.15);
            border: 1px solid rgba(255,255,255,0.3);
            color: #ffffff;
            border-bottom-left-radius: 5px;
        }

        .stTextInput > div > div > input {
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #ffffff33;
            background-color: rgba(255,255,255,0.1);
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialize assistant ---
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = run_pipeline(init_only=True)
    st.session_state["messages"] = []

qa_chain = st.session_state["qa_chain"]

# --- Display Logo + Titles ---
def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

image_path = Path(__file__).parent / "coter_logo.jpeg"
if image_path.exists():
    img_base64 = image_to_base64(image_path)
    st.markdown(f"""
        <div class="center-header">
            <img src="data:image/jpeg;base64,{img_base64}" class="profile-pic">
            <div class='main-title'>Hello dear!</div>
            <div class='sub-title'>I am a chatbot, ask questions and share your data here.</div>
        </div>
    """, unsafe_allow_html=True)

# --- Chat Container ---
chat_container = st.container()

# --- Render Messages ---
with chat_container:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(f"<div class='chat-bubble ai-bubble'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                # Remove user-bubble div, use plain text to avoid double background
                st.markdown(message["content"], unsafe_allow_html=True)

# --- Chat input ---
if prompt := st.chat_input("Type your question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        # Display plain text, no extra div
        st.markdown(prompt, unsafe_allow_html=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("💭 *Thinking...*")

        try:
            result = qa_chain.invoke({"query": prompt})
            response = result["result"] if isinstance(result, dict) else result
        except Exception as e:
            response = f"⚠️ Error: {e}"

        message_placeholder.markdown(f"<div class='chat-bubble ai-bubble'>{response}</div>", unsafe_allow_html=True)

    st.session_state["messages"].append({"role": "assistant", "content": response})
