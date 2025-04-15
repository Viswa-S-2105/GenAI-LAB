# Week 2 Assignments

# Decision Tree and Logistic Regression Projects

This folder contains Python scripts, datasets, and plots for fraud detection and sentiment analysis using machine learning algorithms.

## Contents

### **Python Scripts**
1. **decisionTreeSamp.py**  
   - Demonstrates the implementation of a Decision Tree for fraud detection.
   - Includes sample data processing and visualization.

2. **logisticRegSamp.py**  
   - Implements Logistic Regression for fraud detection and prediction.
   - Provides insights into the logistic decision boundary.

3. **loan_fraud_appDecisionTree.py**  
   - Application-level script for detecting loan fraud using Decision Tree models.
   - Interactive and functional design for real-world applications.

4. **LogisticDecisionModel.py**  
   - Logistic Regression model focused on detecting fraudulent loans.
   - Includes model training, evaluation, and visualization of results.

5. **sentiment_logistic.py**  
   - Logistic Regression script for sentiment analysis.
   - Processes textual data to classify sentiment into positive or negative categories.

---

### **Dataset**
- **loan_fraud_data.csv**  
  - Dataset containing loan application details for fraud detection.
  - Used for training and testing the Decision Tree and Logistic Regression models.

---

### **Plots**
1. **tree_plot.png**  
   - Visualization of the Decision Tree model used in fraud detection.

2. **lr_plot.png**  
   - Logistic Regression decision boundary and metrics plot.

---

## How to Use

1. **Run Fraud Detection Models**:
   - Ensure Python and required libraries are installed (`pandas`, `numpy`, `matplotlib`, `sklearn`, etc.).
   - Execute the scripts with:
     ```bash
     python script_name.py
     ```
   - Replace `script_name.py` with the desired Python file.

2. **Sentiment Analysis**:
   - For `sentiment_logistic.py`, input textual data to analyze sentiment.
   - Model outputs whether the sentiment is positive or negative.

---

## Prerequisites
- Python 3.7 or higher.
- Install required libraries using:
  ```bash
  pip install -r requirements.txt
