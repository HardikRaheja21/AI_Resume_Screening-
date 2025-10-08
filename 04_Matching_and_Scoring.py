#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def compute_keyword_overlap(job_keywords, resume_text):
    resume_low = resume_text.lower()
    matched = sum(1 for kw in job_keywords if kw in resume_low)
    return (matched / len(job_keywords)) * 100 if job_keywords else 0.0

weight_sem = 0.7
weight_kw = 0.3

resumes = {
    "resume1.txt": clean_resume,
}

results = []
for fname, r_text in resumes.items():
    sem_score = util.cos_sim(model.encode(r_text), jd_emb).item() * 100
    kw_score = compute_keyword_overlap(keywords, r_text)
    final_score = weight_sem*sem_score + weight_kw*kw_score
    results.append({
        "filename": fname,
        "sem_score": round(sem_score,2),
        "kw_score": round(kw_score,2),
        "final_score": round(final_score,2)
    })

results_sorted = sorted(results, key=lambda x: x["final_score"], reverse=True)
results_sorted

