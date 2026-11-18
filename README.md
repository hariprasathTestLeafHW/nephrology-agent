# Nephrology Knowledge Assistant

A standalone Streamlit RAG application for general nephrology education. It extracts text from the curated PDFs in `data/knowledge_base`, embeds chunks with Gemini Embeddings, stores the vectors and source metadata in a shared in-memory ChromaDB collection, retrieves relevant excerpts, and asks Gemini Flash-Lite to answer with numbered citations.

## Requirements

- Python 3.10 or newer
- A Google AI Studio API key
- Internet access for Gemini model and embedding requests

The app uses Google's free API tier when available. Model access, quotas, and pricing can change; confirm current terms before use.

## Setup on Windows

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

For local use, put a key from [Google AI Studio](https://aistudio.google.com/apikey) in `.env`. The key is used by the app server and is not written to disk by the app. Keep `.env` private; it is excluded by `.gitignore`. Never paste API keys into chat, screenshots, or source code. If a key is exposed, revoke it and create a replacement.

Run the app:

```powershell
streamlit run app.py --server.address 127.0.0.1
```

The app ships with a shared, read-only corpus in `data/knowledge_base`; visitors cannot upload or modify documents. The app embeds these PDFs when its process starts, preserving each file name and page number for citations. Add new, authorized, clinician-reviewed PDFs to that folder and redeploy/restart the app to update the corpus. Startup embedding uses the configured Gemini key and quota; a process restart rebuilds the in-memory vector store. Check each document's license and update date before adding it.

Text extraction does not include scanned-image OCR. OCR scanned PDFs before adding them. Each file is limited to 20 MB. Indexed text and vectors are held in server memory; PDF files are included in the repository.

## Privacy and medical safety

This is an educational prototype, not a medical device or clinical decision-support system. It does not diagnose, interpret individual patient results, or recommend treatment changes. A clinician should be consulted for personal medical decisions; urgent symptoms need timely professional care.

The vector store is shared by visitors in the same server process; chat history remains per browser session. Startup embedding sends corpus text to Google's API, and question-answer requests send the question and retrieved excerpts. API keys are used by the app server and are not written to disk by this app. Any visitor can use the configured Gemini quota and may incur costs. Google's free-tier terms state that submitted content may be used to improve its products. Do not include protected health information or enter identifiable patient data. Review Google's current [API terms](https://ai.google.dev/gemini-api/terms) and [pricing/data-use details](https://ai.google.dev/gemini-api/docs/pricing) before use.

## Public hosting

The app can be deployed from this repository to [Streamlit Community Cloud](https://share.streamlit.io/). Select branch `main` and `app.py` as the entry point, then add `GEMINI_API_KEY = "your-key"` to the app's Community Cloud **Secrets** settings. Do not put the key in this public repository; a GitHub Actions secret is not available to the running app. A configured server key is required to build the shared corpus and power chat. Anyone with the public link can use your Gemini quota and may incur costs. The app is public and does not require sign-in. Free-tier availability and limits may change.

Answers are grounded in the indexed excerpts and include source references, but retrieval can miss relevant evidence and generated answers can be wrong or incomplete. Verify important claims in the cited source and consult a qualified professional.

## Tests

```powershell
python -m unittest discover -s tests
```
