# Week 5 Assignments

# Multi-Function AI Applications

This folder contains various Python scripts and datasets for applications ranging from medical diagnosis assistance to resume analysis, language translation, news summarization, and chatbot functionality. It demonstrates diverse use cases of AI and machine learning.

## Contents

### **Python Scripts**
1. **app.py**  
   - Main entry point for integrating and executing the various applications in the folder.
   - Provides a unified interface to access functionalities like medical diagnosis, resume analysis, language translation, and more.

2. **medical_diagnosis_assistant.py**  
   - A tool for medical diagnosis assistance based on patient profiles and symptoms.
   - Utilizes the `Disease_symptom_and_patient_profile_dataset.csv` dataset.

3. **news_summarizer.py**  
   - Summarizes lengthy news articles into concise highlights.
   - Aids in quick understanding of key information from articles.

4. **resume_analyser.py**  
   - Analyzes resumes for skill extraction and matches qualifications to job requirements.
   - Handles `.pdf` files like `resumeSample.pdf`.

5. **chatbot.py**  
   - A conversational chatbot that interacts with users for general queries and assistance.
   - Can be further customized for specific domains or applications.

6. **languageTranslator.py**  
   - Translates text between multiple languages.
   - Demonstrates natural language processing capabilities.

---

### **Dataset**
- **Disease_symptom_and_patient_profile_dataset.csv**  
   - Dataset containing information on diseases, symptoms, and patient profiles.
   - Used for training and inference in `medical_diagnosis_assistant.py`.

---

### **Folders**
- **medical_faiss_index/**  
   - Contains pre-built FAISS (Facebook AI Similarity Search) indices for efficient similarity searches in medical datasets or documents.

---

### **Sample Files**
- **resumeSample.pdf**  
   - A sample resume provided for testing and analysis in `resume_analyser.py`.

---

## How to Use

### **1. Run Specific Applications**
- Execute any script by running:
  ```bash
  python script_name.py
