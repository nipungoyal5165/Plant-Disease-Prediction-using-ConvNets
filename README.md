# Plant-Disease-Prediction-using-ConvNets
This project is a Flask-based web application designed to identify plant diseases from leaf images using a trained Convolutional Neural Network (CNN). The application provides users with the disease classification, detailed information about the identified disease, and best practices for plant care.

Features:-

1. Disease Detection: Identify whether a plant leaf shows signs of a specific disease or is healthy.
2. Confidence Levels: Get the confidence score for each prediction.
3. Disease Details: Access additional information about the disease, such as description, commonality, prevention tips, and plant growth requirements.

Setup and Installation:-
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
{  
    "Disease_Name": {  
        "description": "Brief description of the disease.",  
        "commonality": "Regions or conditions where the disease is prevalent.",  
        "prevention": "Tips to prevent the disease.",  
        "plant_info": "General information about the plant."  
    }  
}  

6. Set Up Upload Folder:
The app will store uploaded files in an uploads directory. This folder will be created automatically when you run the app.

7. Run the Application:
Start the Flask app using:
python app.py
 
Access the app in your browser at http://127.0.0.1:5000.
