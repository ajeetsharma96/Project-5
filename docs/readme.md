# Customer Churn Prediction Using Deep Learning

## 📌 Project Overview

Customer churn prediction is an important task for businesses, especially telecom companies. Predicting customer churn helps organizations retain customers and improve customer satisfaction.

This project uses a **Deep Learning Neural Network** to predict whether a customer will churn based on historical customer data.

The project demonstrates:

* Data preprocessing
* Feature encoding
* Deep learning model building
* Model training
* Model evaluation
* Visualization of results 

---

## 🎯 Objectives

The main objectives of this project are:

* Build a deep learning model for churn prediction
* Preprocess customer data effectively
* Train a neural network model
* Evaluate model performance
* Generate performance graphs
* Save trained model
* Prepare project for GitHub submission

---

## 📂 Dataset Information

Dataset Used:

**Telco Customer Churn Dataset**

Dataset File:

```
data/customer_churn.csv
```

Dataset Features Include:

* Gender
* SeniorCitizen
* Partner
* Dependents
* Tenure
* PhoneService
* InternetService
* Contract
* MonthlyCharges
* TotalCharges

Target Variable:

```
Churn
```

Target Values:

* Yes → Customer churned
* No → Customer stayed

Total Records:

≈ 7000 rows

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

1. Loaded dataset using Pandas
2. Removed missing values
3. Encoded categorical variables using LabelEncoder
4. Scaled numerical features using StandardScaler
5. Split dataset into:

```
80% Training Data
20% Testing Data
```

Scaler saved as:

```
outputs/scaler.pkl
```

---

## 🧠 Model Architecture

Deep Learning Model:

Sequential Neural Network

Architecture:

```
Input Layer

Dense Layer (64 neurons)
Activation: ReLU

Dropout Layer (0.3)

Dense Layer (32 neurons)
Activation: ReLU

Dropout Layer (0.2)

Output Layer (1 neuron)
Activation: Sigmoid
```

Compilation Settings:

```
Optimizer: Adam
Loss: Binary Crossentropy
Metrics: Accuracy
```

---

## 📈 Model Training

Training Configuration:

```
Epochs: 50
Batch Size: 16
Validation Split: 0.2
Early Stopping: Enabled
```

Training Output:

Model saved as:

```
outputs/model.h5
```

---

## 📊 Model Evaluation

Evaluation Techniques Used:

* Confusion Matrix
* Accuracy Score
* Precision
* Recall
* F1 Score

Generated Visualizations:

```
outputs/confusion_matrix.png
outputs/accuracy_plot.png
outputs/loss_plot.png
```

These visualizations help analyze model performance.

---

## 📁 Project Structure

```
PROJECT-5/

│
├── data/
│   └── customer_churn.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── outputs/
│   ├── model.h5
│   ├── scaler.pkl
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
├── docs/
│   └── README.md
│
├── venv/
│
├── requirements.txt
├── .gitignore
│
└── README.md
```

---

## ⚙️ Installation Instructions

Step 1 — Clone Repository

```
git clone <repository_link>
cd customer-churn-deep-learning
```

Step 2 — Create Virtual Environment

```
python -m venv venv
```

Step 3 — Activate Virtual Environment

Windows:

```
venv\Scripts\activate
```

Step 4 — Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run Project

Train Model:

```
python src/train.py
```

Evaluate Model:

```
python src/evaluate.py
```

After running, output files will be saved in:

```
outputs/
```

---

## 📊 Results

Model Performance:

```
Training Accuracy ≈ 80%
Validation Accuracy ≈ 80%
```

These results indicate the model performs reasonably well for churn prediction.

---

## 📌 Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* VS Code

---

## 🔍 Key Learnings

Through this project, the following concepts were learned:

* Data preprocessing techniques
* Feature encoding
* Neural network model building
* Model evaluation techniques
* Visualization of model performance
* Machine learning workflow

---

## 🚀 Future Improvements

Possible enhancements:

* Hyperparameter tuning
* Feature engineering
* Use of advanced deep learning models
* Model deployment using APIs
* Integration with real-time systems

---

## 👨‍💻 Author

Name: **Ajit Kumar Bishwkarma**
Role: **Data Science Intern**

Project Type: Internship Machine Learning Project

