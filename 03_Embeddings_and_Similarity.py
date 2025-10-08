from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

resume_emb = model.encode(clean_resume, convert_to_numpy=True)
jd_emb = model.encode(clean_jd, convert_to_numpy=True)

similarity = util.cos_sim(resume_emb, jd_emb).item() * 100
print(f"Semantic match score: {similarity:.2f}%")

