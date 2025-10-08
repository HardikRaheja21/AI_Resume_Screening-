#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install streamlit')
import streamlit as st

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("AI Resume Matcher (Notebook Demo)")

jd_text_input = st.text_area("Paste Job Description")
uploaded_files = st.file_uploader("Upload Resumes", type=["txt","pdf","docx"], accept_multiple_files=True)

if st.button("Match") and jd_text_input and uploaded_files:
    resumes_dict = {}
    for f in uploaded_files:
        resumes_dict[f.name] = extract_text(f)
    st.write("Matching logic goes here (reuse previous cells)")

