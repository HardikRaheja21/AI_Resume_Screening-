#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pdfplumber python-docx')

import os
import pdfplumber
from docx import Document

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")

resume_text = extract_text("data/resumes/resume1.txt")
jd_text = extract_text("data/job_descriptions/sample_jd.txt")
print("Resume snippet:", resume_text[:300])
print("JD snippet:", jd_text[:300])

