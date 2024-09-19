# AI-Driven Cultural Artifact Recognition App

## Overview
This project is an AI-powered web application designed to recognize and classify cultural artifacts from uploaded images. The system uses deep learning algorithms to match artifacts from a dataset and returns relevant details, such as the artifact’s name, history, and significance, sourced from a CSV file (`artifact.csv`). The project is built using **Flask** as the web framework and serves as a practical application of **Data Augmentation in Deep Learning for Computer Vision**.

## Features
- **Image Upload & Recognition**: Users can upload images of artifacts, and the system will match and classify them based on a pre-trained model.
- **Artifact Details**: Once an artifact is recognized, relevant information is fetched from a CSV file and displayed.
- **Augmented Dataset**: The dataset of artifacts includes augmented images for improved classification accuracy.
- **Multilingual Support**: The system provides QR codes and labels in multiple languages for ease of access.
- **Webcam Capture**: Users can capture images via webcam directly on the website for recognition.
- **API Endpoint**: An API endpoint is provided for external systems to classify artifacts by sending images.

## Architecture

- **Flask**: The web framework used for serving the application.
- **Machine Learning**: A deep learning model (likely CNN or ResNet) is used for artifact recognition.
- **Dataset**: The `artifact.csv` file contains metadata about the artifacts (name, description, etc.).
- **HTML/CSS/JavaScript**: For building the frontend with AR and webcam features.
- **Nginx & Gunicorn/Waitress**: For serving the Flask application in production.

## Tech Stack
- **Backend**: Python, Flask, Gunicorn (or Waitress for Windows)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: TensorFlow/Keras or PyTorch (based on your model)
- **Database**: CSV for artifact data storage
- **Deployment**: Nginx, Gunicorn (Linux), Waitress (Windows)
  
## Prerequisites
- Python 3.8+
- Flask
- Gunicorn (for Linux) or Waitress (for Windows)
- TensorFlow/Keras or PyTorch
- Nginx (for Linux server setup)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ArtifactRecognitionApp.git
cd ArtifactRecognitionApp
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup the Dataset

You can download the dataset from the following Google Drive link:

[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1-8Q1RO06Cyu9AuC-7587nMlnSZXz8uNv?usp=sharing)

Once downloaded, place the dataset (CSV and images) in the `data/` folder. Ensure that the `artifact.csv` file is properly formatted with the correct metadata (e.g., artifact name, description, etc.).

### 4. Running the Application Locally

To run the application locally using Flask’s built-in development server:
```bash
python app.py
```

Alternatively, if using **Waitress**:
```bash
waitress-serve --port=8080 app:app
```

### 5. Deploying to a Production Server

#### On Linux (with Gunicorn and Nginx):
- Install Nginx:
  ```bash
  sudo apt-get install nginx
  ```
- Start the Flask app using Gunicorn:
  ```bash
  gunicorn --bind 0.0.0.0:8000 app:app
  ```
- Configure Nginx as a reverse proxy for Gunicorn (see `nginx.conf` for configuration).

#### On Windows (with Waitress):
- Start the Flask app using Waitress:
  ```bash
  waitress-serve --port=8080 app:app
  ```

### 6. Procfile for Deployment (Heroku)
Create a `Procfile` for Heroku deployment:
```
web: waitress-serve --port=$PORT app:app
```

## Example Use Cases

1. **Cultural Artifact Classification**: Upload a picture of an ancient artifact (e.g., statue, pottery) and receive detailed information about its historical significance.
2. **AR Support**: Scan QR codes at museums or historical sites to get additional information on artifacts in multiple languages.
3. **Webcam Integration**: Capture images of artifacts using a webcam for instant recognition.

## Folder Structure
```
.
├── app.py                  # Main Flask application
├── artifact.csv            # CSV file with artifact details
├── data/                   # Dataset directory
├── templates/              # HTML templates
├── static/                 # Static files (CSS, JS)
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku Procfile for deployment
└── README.md               # Project documentation
```

## Future Enhancements
- Implement additional language support.
- Add support for larger datasets by integrating a more robust database (e.g., PostgreSQL).
- Improve the machine learning model with more complex architectures.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
