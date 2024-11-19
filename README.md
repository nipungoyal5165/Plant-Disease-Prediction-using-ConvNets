# Plant-Disease-Prediction-using-ConvNets
This project is a Flask-based web application designed to identify plant diseases from leaf images using a trained Convolutional Neural Network (CNN). The application provides users with the disease classification, detailed information about the identified disease, and best practices for plant care.

Features:-

1. Disease Detection: Identify whether a plant leaf shows signs of a specific disease or is healthy.
2. Confidence Levels: Get the confidence score for each prediction.
3. Disease Details: Access additional information about the disease, such as description, commonality, prevention tips, and plant growth requirements.

## Setup and Installation:-
1. Clone the Repository:
git clone https://github.com/your-username/plant-disease-detection.git  
cd plant-disease-detection  

2. Set Up a Virtual Environment (Recommended):
python -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate  

3. Install Dependencies:
Ensure you have Python 3.8+ installed. Then, run:
pip install -r requirements.txt  

4. Prepare the Model and Class Indices:
- Place your trained model file (plant_disease_detector.keras) in the project directory.
- Include the class_indices.json file, which maps class indices to disease names.

5. Prepare the Disease Details File:
Ensure the disease_info.json file exists in the project directory. It should contain information about each disease in the following format:
```
{  
    "Disease_Name": {  
        "description": "Brief description of the disease.",  
        "commonality": "Regions or conditions where the disease is prevalent.",  
        "prevention": "Tips to prevent the disease.",  
        "plant_info": "General information about the plant."  
    }  
}  
```
7. Set Up Upload Folder:
The app will store uploaded files in an uploads directory. This folder will be created automatically when you run the app.

8. Run the Application:
Start the Flask app using:
python app.py
 
Access the app in your browser at http://127.0.0.1:5000.

## User Instructions
1. Home Page:
- Navigate to the homepage at http://127.0.0.1:5000.
- You'll see a file upload form.

2. Upload a Leaf Image:
- Click "Choose File" and select an image of a plant leaf.
- Ensure the image is in one of the supported formats: .jpg, .jpeg, .png.

3. Predict Disease:
- After uploading the image, click "Predict."
- The app will process the image and return:
  - The predicted disease name.
  - Confidence scores for each disease.


4. View Disease Details:
- If a disease is detected, click the Details button to learn more about the disease, including its description, prevention methods, and plant information.

## File Structure
plant-disease-detection/  
├── app.py                  # Main application script  
├── plant_disease_detector.keras  # Trained model file  
├── class_indices.json      # Mapping of class indices to disease names  
├── disease_info.json       # Detailed disease information  
├── templates/              # HTML templates for the Flask app  
│   ├── index.html          # Homepage template  
│   ├── details.html        # Disease details page template  
├── uploads/                # Directory for uploaded images  
├── static/                 # Directory for static files (e.g., CSS, JS)  
├── requirements.txt        # Python dependencies  
└── README.md               # Project documentation  

## Troubleshooting
- Module Not Found: If you encounter ModuleNotFoundError, ensure all required libraries are installed using pip install -r requirements.txt.
- File Upload Issues: Verify that the uploads/ directory exists and has write permissions.
- Model File Not Found: Confirm that plant_disease_detector.keras is in the root directory.
- Flask App Not Running: Ensure you have activated the virtual environment (if applicable) and that the correct Python version is being used.

## Future Enhancements
- Adding features to highlight disease spots on the leaf image.
- Expanding the model to include additional plant species and diseases.
- Improving the user interface for a better user experience.

> [!TIP]
> Feel free to contribute to this project by forking the repository and submitting pull requests!
