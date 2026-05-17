import streamlit as st
import os
import pandas as pd
from openai import OpenAI
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import nltk
import re
import time

# ---------------- CONFIG ----------------

load_dotenv()

st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🕵️",
    layout="wide"
)

# ---------------- NLTK ----------------

@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

# ---------------- EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ---------------- API KEY ----------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- LIVE WEB SEARCH ----------------

def get_live_context(query: str) -> str:

    try:

        snippets = []

        with DDGS() as ddgs:

            results = ddgs.news(
                keywords=query,
                max_results=10
            )

            for result in results:

                if "body" in result:
                    snippets.append(result["body"])

        return " ".join(snippets)

    except Exception as e:

        return ""

# ---------------- GENERATE RESPONSE ----------------

def generate_response(prompt: str) -> str:

    if not GROQ_API_KEY:
        st.error("Groq API Key missing in .env")
        return ""

    try:

        # Retrieve live news evidence
        live_context = get_live_context(prompt)

        # DEBUG
        st.write("### 🔍 Retrieved Live Evidence")
        st.write(live_context)

        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a real-time factual AI assistant.

IMPORTANT RULES:
- Use ONLY the provided live web evidence.
- Ignore old training knowledge.
- Never mention knowledge cutoff dates.
- Never say 'as of my knowledge cutoff'.
- Give complete factual sentences only.
- Do not give one-word answers.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Answer the question ONLY using the provided live web evidence.

QUESTION:
{prompt}

LIVE WEB EVIDENCE:
{live_context}

FINAL ANSWER:
"""
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        st.error(f"Error generating response: {e}")

        return ""

# ---------------- CLAIM EXTRACTION ----------------

def extract_claims(text: str) -> list:

    sentences = re.split(r'[.!?]+', text)

    claims = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 3
    ]

    return claims

# ---------------- SEARCH CLAIM ----------------

def search_claim(claim: str) -> str:

    return get_live_context(claim)

# ---------------- SIMILARITY ----------------

def compute_similarity(claim: str, evidence: str) -> float:

    if (
        not evidence
        or evidence == "No evidence found."
        or "Error" in evidence
    ):
        return 0.0

    embeddings = embedding_model.encode([claim, evidence])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(score)

# ---------------- VERIFY CLAIM ----------------

def verify_claim(claim: str) -> dict:

    evidence = search_claim(claim)

    similarity = compute_similarity(claim, evidence)

    if similarity > 0.60:
        status = "VERIFIED"
        color = "green"

    elif similarity >= 0.30:
        status = "UNCERTAIN"
        color = "orange"

    else:
        status = "HALLUCINATED"
        color = "red"

    return {
        "claim": claim,
        "evidence": evidence,
        "score": similarity,
        "status": status,
        "color": color
    }

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("ℹ️ About Project")

    st.markdown("""
### LLM Hallucination Detection using RAG

This system:
- retrieves live news evidence
- generates grounded AI responses
- verifies factual claims
- detects hallucinations
""")

# ---------------- MAIN UI ----------------

st.title("🕵️ LLM Hallucination Detection System")

st.markdown(
    "Real-time AI response generation with live verification."
)

if not GROQ_API_KEY:

    st.warning(
        "Please add GROQ_API_KEY in .env file"
    )

user_question = st.text_area(
    "Enter your question:",
    height=120,
    placeholder="e.g. Who is current CM of Tamil Nadu?"
)

if st.button("Generate & Verify"):

    if not user_question.strip():

        st.error("Please enter a question.")

    else:

        with st.spinner("Generating grounded response..."):

            llm_response = generate_response(user_question)

        if llm_response:

            st.subheader("📝 Generated Response")

            st.info(llm_response)

            claims = extract_claims(llm_response)

            if not claims:

                st.warning("No claims extracted.")

            else:

                st.subheader(
                    f"🔍 Verifying {len(claims)} Claims"
                )

                progress_bar = st.progress(0)

                results = []

                total_score = 0.0

                for i, claim in enumerate(claims):

                    result = verify_claim(claim)

                    results.append(result)

                    total_score += result["score"]

                    progress_bar.progress(
                        (i + 1) / len(claims)
                    )

                    time.sleep(0.5)

                avg_score = (
                    total_score / len(claims)
                ) * 100

                st.markdown("---")

                st.markdown(
                    f"## 📊 Overall Trust Score: {avg_score:.1f}%"
                )

                st.markdown(
                    "## Detailed Claim Analysis"
                )

                for res in results:

                    st.markdown(
                        f"""
<div style="
padding:10px;
border-left:5px solid {res['color']};
margin-bottom:10px;
background-color:rgba(128,128,128,0.1);
border-radius:5px;
">
<strong>{res['status']} ({res['score']*100:.1f}%):</strong>
{res['claim']}
</div>
""",
                        unsafe_allow_html=True
                    )

                    with st.expander("View Evidence"):

                        st.write(res["evidence"])

                df = pd.DataFrame(results)

                csv = df.drop(
                    columns=["color"]
                ).to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name="hallucination_report.csv",
                    mime="text/csv"
                )