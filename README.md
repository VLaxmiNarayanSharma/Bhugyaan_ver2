# Land Cover Classification API

This project is a FastAPI-based web application that allows users to upload a any .shp or .geojson format of any admisnstrative area or `.tiff` image with all 12 bands of Sentinel-2 to get a LULC classification aroind the world. It utilizes pre-trained models and a UI also allows users to select a rectangular Region of Interest (ROI) for classification, batch classification for different areas, LULC temporal change dynamics,class probability too on fly without knowing any programming language.

## Prerequisites

Before running the project, ensure that you have the following installed:

- Python 3.9 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository** or download the project folder to your local machine.
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure the following folders exist in the root directory:

uploaded_files: This folder will store the uploaded TIFF files.
classification_methods: Place your pre-trained model .pkl files in this folder.
Ensure that you have your classification models in .pkl format in the classification_methods directory.

Running the Project
Run the FastAPI application:

bash
Copy code
uvicorn main:app --reload --port 1200
Access the application: Open your browser and go to http://127.0.0.1:1200.

Steps to use the application:

1.Upload a .tiff file for classification.
2.Choose a classification method from the dropdown (e.g., SVM, Random Forest).
3.Click on the Submit button.
4.Wait for the classified land cover map to be generated and displayed on the same page.

File Structure
main.py: Contains the FastAPI application logic.
index.html: The template that renders the UI for the app.
classification_methods: Folder where pre-trained model .pkl files are stored.
uploaded_files: Folder where uploaded .tiff files are temporarily stored.
static: Folder that stores the generated classified map images.
License
This project is licensed under the MIT License. Feel free to modify and use it for your own purposes.

This `README.md` provides instructions on how to install, configure, and run your project. It includes information on prerequisites, installation steps, and usage. Make sure to update the repository URL if you’re using a version control system like Git.
