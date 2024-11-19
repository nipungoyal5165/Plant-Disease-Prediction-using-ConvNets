from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import json
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # New imports for metrics

# Initialize Flask app
app = Flask(__name__)

# Load the trained model for plant disease prediction
model = load_model('plant_disease_detector.keras')

# Load class indices to map predicted labels to class names
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    # Returns True if the file has a valid extension, False otherwise
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to prepare the uploaded image for model prediction
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Load and resize image
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to highlight disease spots on the uploaded image
def highlight_disease_spots(img_path):
    img = cv2.imread(img_path)  # Read the image
    img_resized = cv2.resize(img, (128, 128))  # Resize image to fixed dimensions
    
    # Convert image to HSV color space for better color segmentation
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Define HSV range to isolate dark spots typically associated with disease
    lower_bound = np.array([0, 0, 50])   # Lower HSV bound for spots
    upper_bound = np.array([179, 255, 150]) # Upper HSV bound for spots
    
    # Create a mask and highlight disease spots
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    # Detect and draw contours around detected spots
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_img = img_resized.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small spots by area
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(highlighted_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green rectangles

    # Display and save the processed image showing disease spots
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Disease Spots')
    plt.show()

    # Save the highlighted image
    output_path = img_path.replace('.jpg', '_highlighted.jpg')
    cv2.imwrite(output_path, highlighted_img)
    return output_path

# Route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file was submitted with the POST request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Validate and process the uploaded file
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save file to upload directory

        # Prepare the image and get prediction
        img_array = prepare_image(file_path)
        predictions = model.predict(img_array)  # Model returns class probabilities

        # Extract predicted class and confidence levels
        predicted_class_index = np.argmax(predictions)  # Index of highest probability class
        predicted_class = list(class_indices.keys())[predicted_class_index]  # Map to class name
        confidence_levels = predictions[0]  # Probabilities for each class

        # Create a dictionary of confidence levels mapped to class names
        confidence_map = {list(class_indices.keys())[i]: float(confidence) for i, confidence in enumerate(confidence_levels)}

        # Return JSON response with prediction and confidence details
        response = {
            'predicted_class': predicted_class,
            'confidence_levels': confidence_map
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file format'})

# Load additional disease details from a JSON file, if available
disease_info = {}
if os.path.exists('disease_info.json'):
    with open('disease_info.json', 'r') as f:
        disease_info = json.load(f)
else:
    print("Warning: 'disease_info.json' not found. Detailed information will not be available.")

# Route to display detailed information about a specific disease
@app.route('/details/<predicted_class>')
def disease_details(predicted_class):
    # Load disease information from JSON file
    with open('disease_info.json', 'r') as f:
        disease_info = json.load(f)
        
    if predicted_class in disease_info:
        details = disease_info[predicted_class]
        # return render_template('details.html', disease_name=predicted_class, details=details)
    else:
        return "No detailed information available for this disease.", 404

    # Print debug information
    print("Predicted Disease Name:", predicted_class)
    print("Details:", details)

    return render_template('details.html', disease=predicted_class, details=details)

# Main entry point to run the app
if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)  # Start the Flask app in debug mode
