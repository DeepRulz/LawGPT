import os
import time
import base64
import json
import requests
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Legal Assistant")

GROQ_API_KEY = st.secrets.get("groq_api_key", None)

GROQ_MODEL = "openai/gpt-oss-120b"

if True:
    LOGO_PATH = "LawGPTLogo.jpg"
    def file_to_base64(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        ext = os.path.splitext(path)[1].lower()
        mime = "image/jpeg"
        if ext == ".png":
            mime = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif ext == ".svg":
            mime = "image/svg+xml"
        return f"data:{mime};base64,{b64}"

    img_data_uri = file_to_base64(LOGO_PATH)
    if img_data_uri:
        st.markdown(
            f"""
            <style>
            html, body, .stApp, .main, .block-container, .reportview-container, .appview-container, iframe {{
                overflow: visible !important;
                transform: none !important;
            }}
            [class*="stImage"], [class*="element"], [class*="block"], .css-1d391kg, .css-1outpf7 {{
                overflow: visible !important;
                height: auto !important;
                max-height: none !important;
                min-height: 0 !important;
            }}
            .top-left-logo {{
                position: fixed !important;
                top: 16px !important;
                left: 16px !important;
                z-index: 2147483647 !important; /* topmost */
                overflow: visible !important;
                background: #fff !important;
                border-radius: 10px !important;
                padding: 10px !important;
                display: inline-block !important;
                box-shadow: 0 8px 22px rgba(0,0,0,0.30) !important;
                line-height: 0 !important;
            }}
            .top-left-logo img, .top-left-logo img[alt="LawGPT Logo"] {{
                display: block !important;
                width: auto !important;
                height: auto !important;
                max-width: 80px !important;
                max-height: 80px !important;
                object-fit: contain !important;
                -o-object-fit: contain !important;
                border-radius: 6px !important;
                margin: 0 !important;
                padding: 0 !important;
                box-sizing: border-box !important;
                background: transparent !important;
            }}
    
            .top-left-logo, .top-left-logo * {{
                visibility: visible !important;
            }}
            .app-body-padding {{ padding-top: 8px !important; }}
            
            @media (max-width:80px) {{
                .top-left-logo img {{ max-width: 80px !important; max-height: 80px !important; }}
                .app-body-padding {{ padding-top: 20px !important; }}
                .top-left-logo {{ left: 12px !important; top: 12px !important; padding: 8px !important; }}
            }}
            </style>
    
            <div class="top-left-logo">
                <img src="{img_data_uri}" alt="LawGPT Logo" />
            </div>
            <div class="app-body-padding"></div>
            """,
            unsafe_allow_html=True,
        )


st.title("Legal Assistant")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117;
        color: gray;
        text-align: center;
        padding: 8px;
        font-size: 0.85rem;
        z-index: 100;
        border-top: 1px solid #333;
    }
    .footer a { color: #3399ff; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

footer_container = st.container()
with footer_container:
    st.markdown(
        """
        <div class="footer">
            Created by <a href="https://www.linkedin.com/in/deepshah2712/" target="_blank">Deep Shah</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

act_choice = st.selectbox("📚 Choose a Law", ["Right To Information Act,2005", "Code of Civil Procedure,1908","Consumer Protection Act,2019"])

@st.cache_resource
def load_resources(act_name):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_path = f"./Acts/{act_name}/{act_name}.index"
    pkl_path = f"./Acts/{act_name}/{act_name}.pkl"

    if not os.path.exists(index_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Index or pickle NOT found for {act_name}. Expected paths:\n{index_path}\n{pkl_path}")

    index = faiss.read_index(index_path)
    with open(pkl_path, "rb") as f:
        sections = pickle.load(f)
    return model, index, sections

# Wrap load in try/except to show friendly message if resources missing
try:
    model, index, sections = load_resources(act_choice)
except Exception as e:
    st.error(f"Failed to load knowledge base for '{act_choice}': {e}")
    st.stop()

# ------------------- FAISS SEARCH -------------------
def search_faiss(query, k=7):
    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, k)
    results = []
    for i in I[0]:
        try:
            results.append(sections[i])
        except Exception:
            continue
    return results

# ------------------- GROQ CALL -------------------
def ask_groq(query, sections, temperature=0.2, max_tokens=8192):
    """
    Minimal Groq chat-completions call using the OpenAI-compatible endpoint.
    Endpoint used: https://api.groq.com/openai/v1/chat/completions
    (Make sure your GROQ_API_KEY is set in Streamlit secrets.)
    """
    if not GROQ_API_KEY:
        return "❌ Groq API key not found in Streamlit secrets (st.secrets['groq_api_key'])."

    context = "\n\n".join([f"Section {s.get('section','?')} - {s.get('title','')}\n{s.get('text','')}" for s in sections])
    prompt = f"""
You are a legal assistant. Based on the following sections of the {act_choice}, answer the question clearly and concisely.

Question: {query}

Relevant Sections:
{context}

Answer:
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # expected: data["choices"][0]["message"]["content"]
        choices = data.get("choices")
        if not choices:
            return f"❌ Groq returned no choices: {json.dumps(data)[:500]}"
        message = choices[0].get("message", {})
        content = message.get("content") or message.get("content", "") or choices[0].get("text", "")
        return content.strip()
    except requests.exceptions.HTTPError as http_e:
        try:
            d = resp.json()
            return f"❌ Groq API HTTP error {resp.status_code}: {d.get('error') or d}"
        except Exception:
            return f"❌ Groq HTTP error: {http_e}"
    except Exception as e:
        return f"❌ Groq Request error: {e}"

# ------------------- UI: Chat + Interaction -------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# Render previous chat messages
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Ask Your Question")

if query:
    # show user message
    st.chat_message("user").markdown(query)
    st.session_state.chat.append(("user", query))

    # search knowledge base
    with st.spinner("Searching legal knowledge base..."):
        top_sections = search_faiss(query, k=7)

    # if no sections found, show quick message
    if not top_sections:
        answer_text = "I couldn't find relevant sections in the knowledge base. Try rephrasing or pick another Act."
        st.chat_message("assistant").markdown(answer_text)
        st.session_state.chat.append(("ai", answer_text))
    else:
        # call Groq and display animated typing
        with st.chat_message("assistant"):
            placeholder = st.empty()
            # show the sections found (brief) above answer to be transparent
            #briefly = "\n\n".join([f"**{s.get('section','?')} — {s.get('title','')}**" for s in top_sections[:4]])
            #st.markdown("**Searched sections (top results):**\n\n" + briefly)

            # ask Groq
            full_response = ask_groq(query, top_sections)
            # If Groq returned a clear error starting with ❌, fallback to RAG-only summary
            if full_response.startswith("❌"):
                fallback = "Sorry — the model call failed. I searched these sections:\n\n" + "\n\n".join(
                    [f"Section {s.get('section','?')} - {s.get('title','')}" for s in top_sections[:8]]
                )
                st.markdown(f"**Model error:** {full_response}\n\n**Fallback (sections found):**\n{fallback}")
                st.session_state.chat.append(("ai", full_response + "\n\n" + fallback))
            else:
                # stream simple "typing" animation into placeholder
                displayed = ""
                for ch in full_response:
                    displayed += ch
                    placeholder.markdown(displayed)
                    time.sleep(0.006)  # small delay to simulate typing
                st.session_state.chat.append(("ai", full_response))

# ------------------- END -------------------









