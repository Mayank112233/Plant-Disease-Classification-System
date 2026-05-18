Plant Disease Classification System

A deep learning-based image classification project that detects and classifies plant leaf diseases using the PlantVillage dataset and Transfer Learning with EfficientNetB0.

Project Overview

This project uses a Convolutional Neural Network (CNN) with Transfer Learning to identify plant diseases from leaf images.

The model is trained on the PlantVillage dataset and can classify plant leaf images into multiple disease categories.

The project demonstrates:

Deep Learning for Image Classification
Transfer Learning using EfficientNetB0
Data Augmentation
TensorFlow/Keras Model Training
Evaluation and Prediction Pipeline
Features
Multi-class plant disease classification
Uses EfficientNetB0 pre-trained model
Data augmentation for better generalization
Training and validation pipeline
Model checkpoint saving
Prediction support for new leaf images
Easy to extend for deployment
Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
Scikit-learn
Dataset

Dataset used: PlantVillage Dataset

The dataset contains thousands of labeled images of healthy and diseased plant leaves.

Classes include multiple crop diseases such as:

Tomato
Potato
Pepper
Corn
Apple
Grape
Peach
Strawberry

and more.

Model Architecture

The project uses:

EfficientNetB0 (Pre-trained on ImageNet)
GlobalAveragePooling2D
Dense Layer (256 units)
Dropout Layer
Final Softmax Classification Layer
Project Structure
plant-disease-classification/
│
├── dataset/
│   ├── train/
│   └── validation/
│
├── models/
│   └── plant_disease_model.h5
│
├── outputs/
│   ├── accuracy_plot.png
│   └── loss_plot.png
│
├── plant_disease_classifier.py
├── requirements.txt
└── README.md
Installation
1. Clone the Repository
git clone https://github.com/your-username/plant-disease-classification.git
cd plant-disease-classification
2. Create Virtual Environment
conda create -n reviewbot python=3.10
conda activate reviewbot
3. Install Dependencies
pip install -r requirements.txt
Required Libraries

Example requirements.txt

tensorflow
numpy
matplotlib
scikit-learn
pillow
How to Run
python plant_disease_classifier.py
Training Details
Image Size: 224x224
Batch Size: 32
Epochs: 50
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Example Output
Epoch 1/50
accuracy: 0.85
val_accuracy: 0.88
Future Improvements
Deploy using Flask or FastAPI
Convert into a web application
Add real-time disease detection
Integrate medicine recommendation system
Deploy on mobile devices
Learning Outcomes

Through this project, I learned:

Transfer Learning
Image preprocessing
CNN architecture
Model training and evaluation
TensorFlow/Keras workflow
Deep learning project structuring
Author

Mayank Madhukar

BTech CSE (Data Science)

License

This project is open-source and available under the MIT License.
