# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import cv2
import numpy as np

# Load the trained model and CSV file
model = joblib.load('artifact_model.pkl')
artifact_info = pd.read_csv('artifact_info.csv')

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Preprocess uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64)).flatten()  # Resize and flatten the image
    return image.reshape(1, -1)

# Main route to render the frontend
@app.route('/')
def index():
    return render_template('index.html')
# Route for Rajasthan page
@app.route('/rajasthan')
def rajasthan():
    return render_template('rajasthan.html')

# Route for Uttar Pradesh page
@app.route('/uttar_pradesh')
def uttar_pradesh():
    return render_template('uttar_pradesh.html')

# Route for Tamil Nadu page
@app.route('/tamil_nadu')
def tamil_nadu():
    return render_template('tamil_nadu.html')

# Route for Maharashtra page
@app.route('/maharashtra')
def maharashtra():
    return render_template('maharashtra.html')

# Route for Andhra Pradesh page
@app.route('/andhra_pradesh')
def andhra_pradesh():
    return render_template('andhra_pradesh.html')

@app.route('/ar_page')
def ar_page():
    return render_template('ar_page.html')
# Add this route in your Flask app
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')  # Ensure 'webcam.html' is in the templates folder


# API route for image upload and classification
@app.route('/api/upload/', methods=['POST'])
def classify_image():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)

    # Predict the class and get confidence scores (if supported by the model)
    class_probabilities = model.predict_proba(processed_image)  # Check if your model supports this
    class_id = model.predict(processed_image)[0]
    confidence = max(class_probabilities[0]) if class_probabilities is not None else None

    # Debug: Print predicted class_id, confidence score, and available IDs
    print(f"Predicted class_id: {class_id}, Confidence: {confidence}")
    print(f"Available IDs in CSV: {artifact_info['id'].unique()}")

    # Ensure consistent types between class_id and 'id' column
    class_id = str(class_id)
    artifact_info['id'] = artifact_info['id'].astype(str).str.strip()

    # Fetch artifact details from the CSV
    artifact_details = artifact_info[artifact_info['id'] == class_id].to_dict(orient='records')

    # Debugging: Print the fetched artifact details
    print("Fetched artifact details:", artifact_details)

    if artifact_details:
        response = {
            'predicted_class': artifact_details[0].get('artifacts_name', 'N/A'),
            'description': artifact_details[0].get('description', 'N/A'),
            'historical_significance': artifact_details[0].get('historical_significance', 'N/A'),
            'origin': artifact_details[0].get('origin', 'N/A'),
            'confidence': confidence
        }
        # Optionally, add a check for low confidence
        if confidence and confidence < 0.6:  # Example threshold
            response['warning'] = 'Low confidence in prediction.'
        return jsonify(response)
    else:
        print(f"Error: Predicted class_id {class_id} does not match any in CSV")
        return jsonify({'error': f'No matching artifact found for class_id {class_id}'}), 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
