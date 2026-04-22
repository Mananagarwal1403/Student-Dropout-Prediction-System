# 🎓 Student Dropout Prediction System

## 📌 Overview
This project is a **Machine Learning-based system** designed to predict whether a student is likely to drop out or continue their education. It helps educational institutions take early preventive actions using data-driven insights.

---

## 🚀 Features
- 📊 Predict student dropout risk  
- 🤖 Multiple ML models used:
  - Random Forest (Best Model)
  - Logistic Regression
  - Decision Tree  
- ⚖️ Handles imbalanced data using **SMOTE**  
- 📈 Model comparison using graphical visualization  
- 🌐 Interactive web interface using **Streamlit**  

---

## 🧠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Streamlit  
- Matplotlib  

---

## 📂 Project Structure
ML PROJECT
│
├── dataset/
│ ├── student_dataset.csv
│ ├── processed_student_data.csv
│
├── model/
│ └── train_model.py
│
├── app.py
├── model.pkl
├── model_results.csv
├── README.md


---

## ⚙️ Installation

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit imbalanced-learn joblib

##Run the Application
streamlit run app.py

📊 Model Performance
Model	Accuracy
Random Forest	~85%
Logistic Regression	~84%
Decision Tree	~78%

##Workflow
Data Preprocessing
Feature Selection
Train-Test Split
SMOTE (Handling Imbalance)
Model Training
Hyperparameter Tuning (GridSearchCV)
Model Evaluation
Model Comparison
Deployment using Streamlit

## Future Improvements
Add deep learning models
Include psychological and behavioral features
Deploy application on cloud (AWS / Render)
Add user authentication system