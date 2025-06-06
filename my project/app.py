# app.py: Flask app for healthcare chatbot web interface
import logging
from flask import Flask, request, jsonify, render_template
from chatbot_model import get_response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        return "Error loading page", 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        response = get_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return jsonify({'response': 'Server error. Please try again.'}), 500

if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
        logging.info("Flask app started successfully")
    except Exception as e:
        logging.error(f"Error starting Flask app: {e}")