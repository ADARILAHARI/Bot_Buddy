import google.generativeai as genai
import pickle
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Debugging: Print to check if the key is loaded
if GOOGLE_API_KEY:
    print("✅ API Key loaded successfully!")
else:
    print("❌ Error: API Key not found! Please check your .env file.")

# Initialize Gemini API with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Load trained intent classification model and vectorizer
with open("intent_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Intent-based predefined responses
intent_responses = {
    "greeting": "Hello! How can I assist you?",
    "goodbye": "Goodbye! Have a great day!",
    "weather": "I can check the weather for you. What city are you in?",
    "joke": "Here's a joke for you: Why don’t skeletons fight each other? They don’t have the guts!",
}

# Function to predict user intent with confidence
def get_intent(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_probabilities = model.predict_proba(input_vector)  # Get confidence scores
    predicted_intent = model.classes_[predicted_probabilities.argmax()]
    confidence_score = predicted_probabilities.max()

    # If confidence is low, return "unknown"
    if confidence_score < 0.6:  # Adjust threshold based on performance
        return "unknown"
    
    return predicted_intent

# Function to generate AI response (Gemini + intent fallback)
def get_response(user_input):
    intent = get_intent(user_input)

    # If intent has a predefined response, use it
    if intent in intent_responses:
        return intent_responses[intent]

    # If intent is unknown or confidence is low, use Gemini API
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(user_input)
    return response.text  # Return AI-generated response

# Route to render the chatbot page
@app.route("/")
def index():
    return render_template("index.html")  # Ensure index.html is in templates folder

# Route to handle chatbot responses
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    bot_response = get_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)

'''
import google.generativeai as genai
import pickle
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API (Replace with your actual API key)
API_KEY = "AIzaSyAbhcajqobkBdR1J4Mu3smtT5aQCVhwOT4"
genai.configure(api_key=API_KEY)

# Load trained intent classification model and vectorizer
with open("intent_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Intent-based predefined responses
intent_responses = {
    "greeting": "Hello! How can I assist you?",
    "goodbye": "Goodbye! Have a great day!",
    "weather": "I can check the weather for you. What city are you in?",
    "joke": "Here's a joke for you: Why don’t skeletons fight each other? They don’t have the guts!",
}

# Function to predict user intent with confidence
def get_intent(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_probabilities = model.predict_proba(input_vector)  # Get confidence scores
    predicted_intent = model.classes_[predicted_probabilities.argmax()]
    confidence_score = predicted_probabilities.max()

    # If confidence is low, return "unknown"
    if confidence_score < 0.6:  # Adjust threshold based on performance
        return "unknown"
    
    return predicted_intent

# Function to generate AI response (Gemini + intent fallback)
def get_response(user_input):
    intent = get_intent(user_input)

    # If intent has a predefined response, use it
    if intent in intent_responses:
        return intent_responses[intent]

    # If intent is unknown or confidence is low, use Gemini API
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(user_input)
    return response.text  # Return AI-generated response

# Route to render the chatbot page
@app.route("/")
def index():
    return render_template("index.html")  # Ensure index.html is in templates folder

# Route to handle chatbot responses
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    bot_response = get_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)'''