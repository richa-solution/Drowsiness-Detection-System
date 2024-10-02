# Drowsiness Detection System
This project implements a real-time drowsiness detection system using computer vision and machine learning techniques. The system analyzes facial landmarks and eye states to identify drowsiness, providing alerts to enhance driver safety.
## Acknowledgements
Thanks to Kaggle for providing the eye data, and to the communities behind OpenCV, TensorFlow and Streamlit.
 Dataset Link:
```bash
     streamlit run frontend.py
```
## Installation
1. Clone the repository:
```bash
   git clone https://github.com/richa-solution/Drowsiness-Detection_system/edit/main/README.md
```
2. Install Dependencies:
Create Virtual Environment
```bash
    python -m venv venv
```
Activate Environment
```bash
     venv\Scripts\activate
```
3. Install required packages:
Create a requirements.txt file in your project directory with the following content:
   1. opencv-python
   2. numpy
   3. scipy
   4. tensorflow / keras
   5. pillow 
   6. matplotlib  
   7. streamlit
   Then install the dependencies using pip:   
```bash
      pip install -r requirements.txt
```
4. Testing installation:
After installing the dependencies, you can run a sample script to ensure everything is set up correctly:
```bash
     python frontend.py
```
6. Run the Streamlit Application
In your terminal, navigate to the directory where your app.py file is located and run:
```bash
     streamlit run frontend.py
```
    
## Features
- Real-Time Drowsiness Detection: Utilizes computer vision and deep learning to monitor users and detect signs of drowsiness in real-time.
- User-Friendly Interface: Designed with an intuitive interface using Streamlit, making it easy for users to interact with the application.
- Video Input Support: Allows users to upload video files for analysis or use a webcam for live monitoring.
- Data Visualization: Displays graphical representations of detection results, including accuracy and detection time, for user insights.
- Performance Metrics: Offers detailed performance metrics (accuracy, loss) during model training to evaluate the effectiveness of the model.
## Results
This model gave 98% accuracy for drowsiness detection system after training via tensorflow.
## Links
Linkedin Profile
```bash
   https://www.linkedin.com/in/richa-kaushik-a6312a311?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
```
