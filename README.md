# QR Code Authentication System

## 📌 Project Overview  
This project detects counterfeit QR codes by classifying them as **original** or **fake** using machine learning and deep learning models.  

## 📂 Project Structure  
```
QR_Code_Authentication/
│── data_preprocessing.py  # Load & visualize dataset
│── feature_extraction.py  # Extract LBP, edge detection features
│── train_ml.py            # Train SVM & Random Forest models
│── cnn.py           # Train CNN model
│── evaluate.py            # Evaluate models & generate metrics
│── main.py                # Run the entire pipeline
│── requirements.txt       # List dependencies
│── README.md              # Project documentation
```

## 🛠 Installation & Setup  
### **1️⃣ Clone Repository**
```
cd QR_Code_Authentication
```

### **2️⃣ Install Dependencies**  
```
pip install -r requirements.txt
```

### **3️⃣ Run the Pipeline**  
To run the full project:
```
python main.py
```
To train **traditional ML models** (SVM, Random Forest):
```
python train_ml.py
```
To train **CNN model**:
```
python train_cnn.py
```
To evaluate the models:
```
python evaluate.py
```
## Download the trained CNN model from: https://drive.google.com/drive/folders/1btRdz57vhPCZubnpugO1FlR--GATHtil?usp=drive_link

## 🖥️ Model Performance  
- **SVM Accuracy:** 85%  
- **Random Forest Accuracy:** 88%  
- **CNN Accuracy:** 92%  
- **Confusion Matrix & Metrics Included in `report.pdf`**  

## 🔹 Author  
Developed by **Shreya Paul** as part of a QR Code Authentication Assignment.
