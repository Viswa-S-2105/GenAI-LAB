# Week 1 Assignments


# Week 1: Machine Learning and Translation Projects

This folder contains multiple Python scripts and a dataset for exploring machine learning models, data processing, and a translation application.

## Contents

### **Machine Learning Scripts**
1. **housePricePredictForestReg.py**  
   - Implements a Random Forest Regressor for predicting house prices.
   - Uses a structured dataset as input.

2. **housePricePredictForestRegCSV.py**  
   - An extension of the forest regressor script, specifically designed for handling input data from CSV files.

3. **housePricePredictLinearReg.py**  
   - Implements a Linear Regression model for predicting house prices.
   - Demonstrates basic machine learning techniques.

4. **housePricePredictLinearRegCSV.py**  
   - An extension of the linear regression script to support input from CSV files.

### **Dataset**
- **house_prices.csv**  
  - A dataset containing house features and prices.
  - Used for training and evaluating machine learning models.

### **Translation App**
- **maria_translation_app.py**  
  - A Python script leveraging Gradio to create a simple web-based language translation application.
  - Supports translation tasks with a user-friendly interface.

## How to Use

1. **Run Machine Learning Scripts**:
   - Ensure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`, etc.).
   - Use the command:
     ```bash
     python script_name.py
     ```
   - Replace `script_name.py` with the specific file you want to execute.

2. **Use the Translation App**:
   - Install dependencies (`gradio`, etc.) if not already installed.
   - Launch the application:
     ```bash
     python maria_translation_app.py
     ```
   - Access the web interface to test language translations.

## Prerequisites
- Python 3.7 or above.
- Install required libraries using:
  ```bash
  pip install -r requirements.txt
