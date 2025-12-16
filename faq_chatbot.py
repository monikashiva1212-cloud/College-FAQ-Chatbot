import streamlit as st
import nltk
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# -----------------------------
# COLLEGE FAQ DATA
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

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

processed_questions = [preprocess(q) for q in questions]

# -----------------------------
# VECTORIZE
# -----------------------------
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = np.argmax(similarity)

    if similarity[0][best_match] < 0.2:
        return "Sorry, I didn't understand that."
    return answers[best_match]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎓 College FAQ Chatbot")
st.write("Ask any question related to the college")

user_input = st.text_input("Your Question:")

if st.button("Ask"):
    if user_input:
        response = chatbot_response(user_input)
        st.success(response)