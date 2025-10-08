#
# STEP A: PASTE ALL YOUR IMPORTS HERE
#
import streamlit as st
import os
import pdfplumber
from docx import Document
import spacy
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- 1. Functions from your parsing script ---
def extract_text(file_path):
    # This function now needs to handle Streamlit's file-like object
    # Make sure to read the file content before passing it to the libraries
    file_extension = os.path.splitext(file_path.name)[1].lower()
    if file_extension == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_extension == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_extension == ".txt":
        return file_path.read().decode("utf-8")
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

# --- 2. Functions from your preprocessing script ---
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc 
              if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num]
    return " ".join(tokens)

def extract_keywords(text, top_k=20):
    doc = nlp(text)
    candidates = [chunk.text.lower() for chunk in doc.noun_chunks]
    candidates += [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN","PROPN") and not token.is_stop]
    counts = Counter(candidates)
    return [k for k,_ in counts.most_common(top_k)]

# --- 3. Function from your matching script ---
def compute_keyword_overlap(job_keywords, resume_text):
    resume_low = resume_text.lower()
    matched = sum(1 for kw in job_keywords if kw in resume_low)
    return (matched / len(job_keywords)) * 100 if job_keywords else 0.0

# --- 4. Load the Sentence Transformer Model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

#
# STEP B: YOUR STREAMLIT INTERFACE CODE
#
st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("AI Resume Matcher")

jd_text_input = st.text_area("Paste Job Description")
uploaded_files = st.file_uploader("Upload Resumes", type=["txt","pdf","docx"], accept_multiple_files=True)

if st.button("Match") and jd_text_input and uploaded_files:
    # Now you can call the functions because they are in this file
    
    # Process the Job Description
    clean_jd = clean_text(jd_text_input)
    jd_keywords = extract_keywords(jd_text_input)
    jd_emb = model.encode(clean_jd)
    
    results = []
    st.write("--- Matching Results ---")

    for resume_file in uploaded_files:
        with st.spinner(f"Processing {resume_file.name}..."):
            # 1. Extract and clean resume text
            resume_text = extract_text(resume_file)
            clean_resume = clean_text(resume_text)
            
            # 2. Calculate scores
            sem_score = util.cos_sim(model.encode(clean_resume), jd_emb).item() * 100
            kw_score = compute_keyword_overlap(jd_keywords, clean_resume)
            
            # 3. Compute final weighted score
            weight_sem = 0.7
            weight_kw = 0.3
            final_score = weight_sem * sem_score + weight_kw * kw_score
            
            results.append({
                "filename": resume_file.name,
                "final_score": round(final_score, 2),
                "sem_score": round(sem_score, 2),
                "kw_score": round(kw_score, 2)
            })

    # Sort and display results
    results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    for rank, result in enumerate(results_sorted, 1):
        st.subheader(f"Rank {rank}: {result['filename']}")
        st.progress(int(result['final_score']))
        st.write(f"**Overall Match Score:** {result['final_score']}%")
        st.write(f"Semantic Match: {result['sem_score']}% | Keyword Match: {result['kw_score']}%")
