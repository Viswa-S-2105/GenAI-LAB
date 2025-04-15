import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
import docx
import json
from io import BytesIO
from pathlib import Path
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Resume Analyzer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_bytes):
    text = ""
    try:
        doc = docx.Document(docx_bytes)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
    return text

# Function to extract text from TXT
def extract_text_from_txt(txt_bytes):
    try:
        return txt_bytes.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error extracting text from TXT: {e}")
        return ""

# Function to extract text based on file type
def extract_text(file):
    # Get file extension
    file_extension = Path(file.name).suffix.lower()
    
    # Create a temporary file to handle the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    # Process based on file type
    if file_extension == '.pdf':
        with open(tmp_path, 'rb') as f:
            text = extract_text_from_pdf(f)
    elif file_extension == '.docx':
        with open(tmp_path, 'rb') as f:
            text = extract_text_from_docx(f)
    elif file_extension == '.txt':
        with open(tmp_path, 'rb') as f:
            text = extract_text_from_txt(f)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        text = ""
    
    # Clean up the temporary file
    os.unlink(tmp_path)
    return text

# Function to analyze resume
def analyze_resume(model, resume_text, job_description=None):
    """
    Analyze a resume using the Gemini model and return structured insights.
    """
    # Basic prompt if no job description
    if not job_description:
        prompt = f"""
        Analyze the following resume and provide detailed insights:
        
        Resume:
        {resume_text}
        
        Please provide the following in a structured format:
        1. Contact information
        2. Skills (technical and soft skills)
        3. Experience summary (years of experience and key roles)
        4. Education details
        5. Key strengths
        6. Areas for improvement
        7. Specific recommendations to enhance the resume
        
        Format your response so it can be easily displayed in a web application.
        """
    else:
        # Enhanced prompt with job matching
        prompt = f"""
        Analyze the following resume against the provided job description:
        
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Please provide the following in a structured format:
        1. Contact information
        2. Skills (technical and soft skills)
        3. Experience summary (years of experience and key roles)
        4. Education details
        5. Key strengths
        6. Skills matching job requirements (percentage match and missing skills)
        7. Experience relevance to the job (scale 1-10 with explanation)
        8. Areas for improvement
        9. Specific recommendations to better align the resume with the job description
        10. Overall match score (percentage with explanation)
        
        Format your response so it can be easily displayed in a web application.
        """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return f"Error: {str(e)}"

def main():
    # Sidebar for API configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input("Enter Google API Key", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
    
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This Resume Analyzer uses Google's Gemini 1.5 Pro model to:
        - Extract skills and experience
        - Identify strengths and gaps
        - Provide personalized recommendations
        - Match resumes to job descriptions
        """)
    
    # Main content
    st.title("📄 Resume Analyzer AI")
    st.markdown("Upload your resume and optional job description to get personalized insights and recommendations.")
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT")
    
    with col2:
        job_file = st.file_uploader("Upload Job Description (Optional)", type=["pdf", "docx", "txt"], help="Adds job matching analysis")
    
    # Process button
    analyze_button = st.button("Analyze Resume", type="primary", disabled=not (resume_file and api_key))
    
    if analyze_button and resume_file and api_key:
        # Show progress
        with st.spinner("Processing your resume..."):
            # Extract text from resume
            resume_text = extract_text(resume_file)
            
            # Extract text from job description if provided
            job_text = None
            if job_file:
                job_text = extract_text(job_file)
            
            # Load model
            model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            
            # Analyze
            analysis = analyze_resume(model, resume_text, job_text)
        
        # Display results
        st.success("Analysis complete!")
        
        # Display in tabs
        tab1, tab2 = st.tabs(["Analysis Results", "Raw Resume Text"])
        
        with tab1:
            st.markdown("## Resume Analysis")
            st.markdown(analysis)
        
        with tab2:
            st.markdown("## Extracted Resume Text")
            st.text_area("", resume_text, height=400)
            
            if job_text:
                st.markdown("## Extracted Job Description")
                st.text_area("", job_text, height=300)

if __name__ == "__main__":
    main()