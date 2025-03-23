# Implementation-of-chatbot-NLP
Training for the project on Implementation of Chatbot using NLP

This project is an interactive AI chatbot built using Natural Language Processing (NLP) and Logistic Regression, with a Streamlit-based UI for a seamless user experience. The chatbot processes user input using TF-IDF vectorization, classifies the intent with Logistic Regression, and provides relevant responses. Additionally, it maintains a conversation history by logging interactions in a CSV file.

Features:
The chatbot utilizes TF-IDF vectorization for text processing and Logistic Regression for intent classification. It supports an intuitive Streamlit-based user interface where users can chat and view their conversation history. The chatbot also logs conversations for future reference and analysis.

Project Structure:
The project consists of several key files:

app.py – The main chatbot script that handles user interaction.

intent.json – A dataset containing predefined intents, patterns, and responses.

chat_log.csv – A file that logs user inputs and chatbot responses.

requirements.txt – The list of dependencies required to run the chatbot.

nltk_data/ – The folder containing necessary NLP datasets.

 Future Improvements:
In the future, this chatbot can be enhanced by integrating deep learning models like RNNs or Transformers for improved intent recognition. Additionally, real-time response updates and an advanced UI can be implemented to enhance the user experience.
