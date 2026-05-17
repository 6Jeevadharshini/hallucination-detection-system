# LLM Hallucination Detection and Fact Verification System

This application detects possible hallucinations in Large Language Model (LLM) responses by verifying factual claims using real web evidence and semantic similarity scoring.

## Architecture

1. **User Input:** A Streamlit interface where users can enter a question or prompt.
2. **LLM Generation:** The system uses the Google Gemini API to generate a detailed response to the query.
3. **Claim Extraction:** The generated response is broken down into atomic factual claims using NLTK sentence tokenization.
4. **Web Search Verification:** Each extracted claim is searched using SerpAPI to retrieve top snippets from Google Search as "evidence."
5. **Semantic Similarity:** A `sentence-transformers` model (`all-MiniLM-L6-v2`) computes the cosine similarity between the original claim and the web evidence.
6. **Trust Score:** Claims are assigned a status based on their similarity score:
   - **VERIFIED:** > 75%
   - **UNCERTAIN:** 45% - 75%
   - **HALLUCINATED:** < 45%

## Installation

1. Clone this repository or navigate to your project directory.
2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Setup

You will need API keys for Google Gemini and SerpAPI.

1. Rename the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).
3. Get a SerpAPI key from [SerpAPI](https://serpapi.com/).
4. Add the keys to your `.env` file:
   ```env
   GEMINI_API_KEY="your_actual_gemini_key"
   SERPAPI_KEY="your_actual_serpapi_key"
   ```

## Running the Application

Execute the following command in your terminal:

```bash
streamlit run app.py
```

## Future Enhancements

- Fine-tune claim extraction with LLM-based atomic fact formulation.
- Integrate additional search providers or knowledge graphs (e.g., Wikipedia API).
- Support document upload for fact-checking pre-written documents.
- Allow swapping out embedding models and thresholds dynamically from the UI.
