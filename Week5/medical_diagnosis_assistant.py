import os
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Load environment variables
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class MedicalDiagnosisAssistant:
    def __init__(self):
        self.dataset_path = "Disease_symptom_and_patient_profile_dataset.csv"
        self.disclaimer = """
IMPORTANT MEDICAL DISCLAIMER:
The information provided is for educational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
Never disregard professional medical advice or delay in seeking it because of something you have read here.
"""
        self._initialize_vector_store()
        self._setup_language_model()

    def _initialize_vector_store(self):
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vectorstore = FAISS.load_local("medical_faiss_index", self.embeddings)
            print("✅ Loaded existing vector store.")
        except Exception as e:
            print(f"⚠️ Creating new vector store due to: {e}")
            self._create_vector_store()

    def _create_vector_store(self):
        df = pd.read_csv(self.dataset_path)
        df.fillna("", inplace=True)

        documents = []
        for _, row in df.iterrows():
            profile = f"Disease: {row['Disease']}, Fever: {row['Fever']}, Cough: {row['Cough']}, Fatigue: {row['Fatigue']}, Difficulty Breathing: {row['Difficulty Breathing']}, Age: {row['Age']}, Gender: {row['Gender']}, Blood Pressure: {row['Blood Pressure']}, Cholesterol Level: {row['Cholesterol Level']}"
            documents.append(profile)

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = FAISS.from_texts(documents, self.embeddings)
        self.vectorstore.save_local("medical_faiss_index")

    def _setup_language_model(self):
        prompt_template = """
You are a medical diagnosis assistant. Your role is to analyze symptoms and suggest possible diagnoses.

IMPORTANT: Always include appropriate disclaimers and never present your analysis as definitive medical advice.

Given the following symptoms and patient profile:
{symptoms}

And based on the following medical information:
{context}

Please provide:
1. A list of possible conditions that might match these symptoms (maximum 3)
2. For each condition, explain why it might match the symptoms
3. Mention any red flags that might require immediate medical attention
4. Suggest what kind of medical professional might be appropriate to consult

Remember: You are NOT replacing professional medical advice. Your suggestions are for informational purposes only.
"""

        prompt = PromptTemplate(
            input_variables=["symptoms", "context"],
            template=prompt_template,
        )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )

    def get_diagnosis(self, symptoms: str) -> str:
        try:
            result = self.qa_chain.invoke({"symptoms": symptoms})
            return result["result"] + "\n\n" + self.disclaimer
        except Exception as e:
            return f"Error generating diagnosis: {str(e)}\n\n{self.disclaimer}"


def main():
    assistant = MedicalDiagnosisAssistant()

    print("Medical Diagnosis Assistant")
    print("==========================")
    print("Enter your symptoms and profile (e.g., 'fever, 25, male') or type 'exit' to quit.")

    while True:
        user_input = input("\nSymptoms and Profile: ").strip()
        if user_input.lower() == "exit":
            print("Thank you for using the Medical Diagnosis Assistant. Stay healthy!")
            break
        if not user_input:
            print("Please enter some symptoms and profile info.")
            continue

        response = assistant.get_diagnosis(user_input)
        print("\n" + response)


if __name__ == "__main__":
    main()
