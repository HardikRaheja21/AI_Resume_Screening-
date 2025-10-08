#
# --- IMPORTS ---
#
import streamlit as st
import os
import pdfplumber
from docx import Document
import spacy
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np
import subprocess # <-- Make sure this import is added

#
# --- MODEL LOADING ---
#

# Use st.cache_resource for models to load them only once
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    """Loads the spaCy model, downloading if necessary."""
    try:
        # Try to load the model directly
        nlp = spacy.load(model_name)
    except OSError:
        # If it fails, the model is not found. Download it.
        st.info(f"First-time setup: Downloading spaCy model '{model_name}'. This may take a moment...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            nlp = spacy.load(model_name)
        except subprocess.CalledProcessError as e:
            st.error(f"Error downloading spaCy model: {e}")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during model loading: {e}")
            st.stop()
    return nlp

@st.cache_resource
def load_sentence_model():
    """Loads the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Assign the loaded models to variables
nlp = load_spacy_model()
model = load_sentence_model()


#
# --- HELPER FUNCTIONS ---
#

def extract_text(uploaded_file):
    """Extracts text from an uploaded file (PDF, DOCX, or TXT)."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension == ".pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_extension == ".docx":
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_extension == ".txt":
        return uploaded_file.read().decode("utf-8")
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

def clean_text(text):
    """Cleans the text by lemmatizing and removing stop words and punctuation."""
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num]
    return " ".join(tokens)

def extract_keywords(text, top_k=20):
    """Extracts keywords from text using noun chunks and POS tagging."""
    doc = nlp(text)
    candidates = [chunk.text.lower() for chunk in doc.noun_chunks]
    candidates += [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
    counts = Counter(candidates)
    return [k for k, _ in counts.most_common(top_k)]

def compute_keyword_overlap(job_keywords, resume_text):
    """Computes the percentage of job keywords present in the resume."""
    resume_low = resume_text.lower()
    matched = sum(1 for kw in job_keywords if kw in resume_low)
    return (matched / len(job_keywords)) * 100 if job_keywords else 0.0

#
# --- STREAMLIT UI ---
#

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("🤖 AI Resume Matcher")

# --- Inputs ---
jd_text_input = st.text_area("Paste Job Description Below:", height=250)
uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT):", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# --- Matching Logic ---
if st.button("✨ Match Resumes"):
    if not jd_text_input or not uploaded_files:
        st.warning("Please paste a job description and upload at least one resume.")
    else:
        # Process the Job Description
        with st.spinner("Analyzing Job Description..."):
            clean_jd = clean_text(jd_text_input)
            jd_keywords = extract_keywords(jd_text_input)
            jd_emb = model.encode(clean_jd)
        
        results = []
        st.write("---")
        st.header("📊 Matching Results")

        for resume_file in uploaded_files:
            with st.spinner(f"Processing {resume_file.name}..."):
                resume_text = extract_text(resume_file)
                if not resume_text.strip():
                    st.warning(f"Could not extract text from {resume_file.name}. Skipping.")
                    continue

                clean_resume = clean_text(resume_text)
                resume_emb = model.encode(clean_resume)
                sem_score = util.cos_sim(resume_emb, jd_emb).item() * 100
                kw_score = compute_keyword_overlap(jd_keywords, clean_resume)
                
                weight_sem = 0.7
                weight_kw = 0.3
                final_score = (weight_sem * sem_score) + (weight_kw * kw_score)
                
                results.append({
                    "filename": resume_file.name,
                    "final_score": final_score,
                    "sem_score": sem_score,
                    "kw_score": kw_score
                })

        # Sort and display results
        results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        for rank, result in enumerate(results_sorted, 1):
            st.subheader(f"Rank {rank}: {result['filename']}")
            st.progress(int(result['final_score']))
            st.write(f"**📈 Overall Match Score:** {result['final_score']:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Semantic Match", value=f"{result['sem_score']:.2f}%")
            with col2:
                st.metric(label="Keyword Match", value=f"{result['kw_score']:.2f}%")
            st.write("---")
