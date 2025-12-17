import streamlit as st
import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="College FAQ Bot",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# -----------------------------
# CUSTOM CSS (DARK MODE + WHATSAPP)
# -----------------------------
st.markdown(f"""
<style>
body {{
    background: {"#0b141a" if st.session_state.dark_mode else "linear-gradient(135deg, #e3f2fd, #f8fbff)"};
}}

.main-card {{
    background-color: {"#111b21" if st.session_state.dark_mode else "#ffffff"};
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}}

.title-text {{
    font-size: 36px;
    font-weight: bold;
    color: {"#e9edef" if st.session_state.dark_mode else "#0d47a1"};
    text-align: center;
}}

.subtitle-text {{
    text-align: center;
    color: {"#aebac1" if st.session_state.dark_mode else "#455a64"};
    font-size: 18px;
    margin-bottom: 25px;
}}

.stButton>button {{
    background: {"#25d366" if st.session_state.dark_mode else "linear-gradient(135deg, #1e88e5, #42a5f5)"};
    color: white;
    font-size: 17px;
    border-radius: 14px;
    padding: 8px 24px;
    border: none;
}}

.stTextInput>div>div>input {{
    border-radius: 20px;
    padding: 14px;
    border: none;
    background-color: {"#202c33" if st.session_state.dark_mode else "#e3f2fd"};
    color: {"white" if st.session_state.dark_mode else "black"};
}}

.chat-container {{
    margin-top: 20px;
}}

.user-bubble {{
    background-color: {"#005c4b" if st.session_state.dark_mode else "#c8e6c9"};
    color: white;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    margin-left: auto;
    margin-bottom: 12px;
    font-size: 16px;
}}

.bot-bubble {{
    background-color: {"#202c33" if st.session_state.dark_mode else "#e1f5fe"};
    color: {"#e9edef" if st.session_state.dark_mode else "#0d47a1"};
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    margin-bottom: 12px;
    font-size: 16px;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FAQ DATA
# -----------------------------
faqs = {
    "What courses are offered in the college?":
        "Our college offers Engineering, Arts, Science, and Management courses.",

    "How can I apply for admission?":
        "You can apply online through the college official website or visit the admission office.",

    "Is hostel facility available?":
        "Yes, hostel facilities are available for both boys and girls.",

    "Does the college provide placement support?":
        "Yes, our college has an active placement cell that supports students.",

    "What are the college working hours?":
        "The college works from 9:00 AM to 4:30 PM, Monday to Friday.",

    "Is scholarship available?":
        "Yes, scholarships are available for merit and economically weaker students."
}

questions = list(faqs.keys())
answers = list(faqs.values())

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform([preprocess(q) for q in questions])

def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = np.argmax(similarity)

    if similarity[0][best_match] < 0.2:
        return "🤔 Sorry, I didn’t understand that."
    return answers[best_match]

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "home":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<div class="title-text">🎓 College FAQ Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Welcome! Get instant answers 💙</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📘 Enter FAQ"):
            st.session_state.page = "faq"
    with col2:
        if st.button("❌ Exit"):
            st.stop()

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FAQ PAGE
# -----------------------------
elif st.session_state.page == "faq":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.toggle("🌙 Dark Mode", key="dark_mode")

    st.markdown('<div class="title-text">💬 Ask Your Question</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">WhatsApp style chat 😊</div>', unsafe_allow_html=True)

    user_input = st.text_input("Type your question")

    if st.button("Send"):
        if user_input:
            response = chatbot_response(user_input)

            st.markdown(f"""
            <div class="chat-container">
                <div class="user-bubble">{user_input}</div>
                <div class="bot-bubble">🤖 {response}</div>
            </div>
            """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
    with col2:
        if st.button("❌ Exit"):
            st.stop()

    st.markdown('</div>', unsafe_allow_html=True)
