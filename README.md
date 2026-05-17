# 🕵️ Real-Time LLM Hallucination Detection using RAG

This project detects hallucinations in Large Language Model (LLM) responses using a Retrieval-Augmented Generation (RAG) pipeline with real-time web evidence retrieval and semantic verification.

---

# 🚀 Features

- Real-time AI response generation
- Retrieval-Augmented Generation (RAG)
- Live web evidence retrieval using DuckDuckGo
- Hallucination detection
- Claim extraction and verification
- Semantic similarity scoring
- Trust score generation
- CSV report download
- Interactive Streamlit UI

---

# 🧠 Tech Stack

## Frontend

- Streamlit

## Backend

- Python
- Groq API (LLM)
- DuckDuckGo Search
- Sentence Transformers
- Scikit-learn
- NLTK

---

# ⚙️ System Architecture

1. User enters a question in the Streamlit interface.
2. DuckDuckGo retrieves real-time web evidence.
3. Retrieved evidence is passed to the Groq LLM.
4. The LLM generates a grounded response using live evidence.
5. Claims are extracted from the response.
6. Each claim is verified using semantic similarity.
7. Claims are labeled as:
   - VERIFIED
   - UNCERTAIN
   - HALLUCINATED
8. Overall trust score is generated.

---

# 📊 Verification Logic

The system uses cosine similarity between:

- generated claim
- retrieved live evidence

### Thresholds

| Similarity Score | Status       |
| ---------------- | ------------ |
| > 60%            | VERIFIED     |
| 30% - 60%        | UNCERTAIN    |
| < 30%            | HALLUCINATED |

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/6Jeevadharshini/hallucination-detection-system.git
```
