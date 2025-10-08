import spacy
from collections import Counter

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

clean_resume = clean_text(resume_text)
clean_jd = clean_text(jd_text)
keywords = extract_keywords(jd_text)
print("Keywords:", keywords)

