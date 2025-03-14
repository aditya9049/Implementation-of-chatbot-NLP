import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure SSL context is set correctly
ssl._create_default_https_context = ssl._create_unverified_context

# Set up NLTK data path
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intent.json")
if not os.path.exists(file_path):
    st.error("Error: intents.json file not found!")
    st.stop()

with open(file_path, "r", encoding='utf-8') as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to generate chatbot response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

# Ensure chat log file exists
chat_log_file = 'chat_log.csv'
if not os.path.exists(chat_log_file):
    with open(chat_log_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

# Custom CSS for better UI
st.markdown(
    """
    <style>
        .stApp { background-color: #f8f9fa; }
        .chat-container { max-width: 800px; margin: auto; }
        .user-message { background-color: #dcf8c6; padding: 10px; border-radius: 10px; width: fit-content; }
        .bot-message { background-color: #e2e2e2; padding: 10px; border-radius: 10px; width: fit-content; }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App
def main():
    st.sidebar.title("💬 Chatbot Menu")
    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "Chat":
        st.title("🤖 AI Chatbot")
        st.markdown("### Welcome! Ask me anything.")

        # Chat interface using columns
        user_input = st.text_input("You:", key="input")
        col1, col2 = st.columns([5, 1])
        with col2:
            send_button = st.button("Send")

        if send_button and user_input:
            response = chatbot(user_input)

            # Display chat in styled messages
            st.markdown(f"<div class='chat-container'><p class='user-message'><b>You:</b> {user_input}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-container'><p class='bot-message'><b>Bot:</b> {response}</p></div>", unsafe_allow_html=True)

            # Save conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(chat_log_file, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Auto-scroll down
            st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

    elif choice == "Conversation History":
        st.title("📜 Conversation History")
        if os.path.exists(chat_log_file):
            with open(chat_log_file, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.markdown(f"<p class='user-message'><b>User:</b> {row[0]}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='bot-message'><b>Bot:</b> {row[1]}</p>", unsafe_allow_html=True)
                    st.text(f"🕒 Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.warning("No conversation history available.")

    elif choice == "About":
        st.title("ℹ️ About")
        st.markdown("""
        **🤖 AI Chatbot using NLP & Logistic Regression**
        
        - Uses **TF-IDF Vectorization** for text processing.
        - Classifies user intents using **Logistic Regression**.
        - Built with **Streamlit** for a clean and interactive UI.
        
        **✨ Future Improvements:**
        - Integrating deep learning models (RNNs, Transformers).
        - More advanced NLP techniques.
        - Improved UI with real-time chat updates.
        """)

if __name__ == '__main__':
    main()
