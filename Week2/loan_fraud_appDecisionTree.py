import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import gradio as gr

# 1. Load dataset
df = pd.read_csv("loan_fraud_data.csv")

# 2. Features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# 3. Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 4. Prediction function
def predict_fraud(loan_amount, loan_duration_months, credit_score, income, age, num_previous_loans):
    input_data = pd.DataFrame([[
        loan_amount, loan_duration_months, credit_score, 
        income, age, num_previous_loans
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]
    
    label = "FRAUDULENT" if prediction == 1 else "LEGIT"
    return f"Prediction: {label} (Confidence: {probability:.2f})"

# 5. Gradio UI
interface = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Duration (months)"),
        gr.Number(label="Credit Score"),
        gr.Number(label="Income"),
        gr.Number(label="Age"),
        gr.Number(label="Number of Previous Loans"),
    ],
    outputs="text",
    title="Loan Fraud Detection",
    description="Enter loan details to predict if the application is fraudulent or legit."
)

# 6. Launch
interface.launch()
