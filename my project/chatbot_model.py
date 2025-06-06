# chatbot_model.py: Model training and prediction for healthcare chatbot
import json
import logging
import random
from preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
try:
    with open('intents.json', 'r') as file:
        data = json.load(file)
    logging.info("Dataset loaded successfully")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    exit(1)

# Prepare training data
X = []
y = []
try:
    for intent in data['intents']:
        for pattern in intent['patterns']:
            X.append(preprocess_text(pattern))
            y.append(intent['tag'])
    logging.info("Training data prepared")
except Exception as e:
    logging.error(f"Error preparing training data: {e}")
    exit(1)

# Train model
try:
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X, y)
    logging.info("Model trained successfully")
except Exception as e:
    logging.error(f"Error training model: {e}")
    exit(1)

# Predict intent and get response
def get_response(user_input):
    try:
        if not user_input.strip():
            return "Please enter a valid query."
        processed_input = preprocess_text(user_input)
        predicted_intent = model.predict([processed_input])[0]
        for intent in data['intents']:
            if intent['tag'] == predicted_intent:
                return random.choice(intent['responses'])
        return "Sorry, I didn't understand. Can you rephrase?"
    except Exception as e:
        logging.error(f"Error predicting intent: {e}")
        return "An error occurred. Please try again."
