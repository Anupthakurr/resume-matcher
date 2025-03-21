import streamlit as st
import pickle
import docx
import PyPDF2
import re
import requests
import os


# Load pre-trained model and vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Function to extract text from files
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''.join([page.extract_text() for page in pdf_reader.pages])
    return text

def extract_text_from_docx(file):
    return '\n'.join([paragraph.text for paragraph in docx.Document(file).paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Streamlit UI
def main():
    st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
        <style>   
        .stFileUploader label {
    color: #6A1B9A; /* Changed to dark purple */
    font-size: 18px;
    font-weight: bold;
}
            .upload-box {
         border: 2px dashed #0288D1;
         padding: 5px;
         border-radius: 10px;
         background-color: #DFFFD6; /* Changed to light green */
         color: #0277BD;
         text-align: center;
         font-size: 18px;
         font-weight: bold;
         box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.1);
}
            .result {
                font-size: 24px;
                color: #0D47A1;
                text-align: center;
                font-weight: bold;
                padding: 5px;
                background-color: #E1F5FE;
                border-radius: 10px;
                margin-top: 5px;
                box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
            }
            .button-style {
                background: linear-gradient(to right, #0288D1, #01579B);
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 18px;
                border-radius: 8px;
                cursor: pointer;
                transition: 0.3s;
                font-weight: bold;
            }
            .button-style:hover {
                background: linear-gradient(to right, #01579B, #0288D1);
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    div.stFileUploader {
        background-color: #E3F2FD !important; /* Light Blue Background */
        border-radius: 10px !important;
        color: #0277BD !important; /* Text Color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown(
    """
    <style>
    /* Reduce top margin for Streamlit container */
    .block-container {
        padding-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown("<h1 style='text-align: center; color: #01579B;'>📄 Resume Category Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #0277BD;'>Upload a resume (PDF, DOCX, TXT) and get instant job category predictions!</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], help="Drag and drop your file here")
    
    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.markdown("<div class='upload-box'>✔️ File uploaded successfully!</div>", unsafe_allow_html=True)
            
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=250)
            
            category = pred(resume_text)
            st.markdown(f"<div class='result'>📝 Predicted Category: <b>{category}</b></div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

