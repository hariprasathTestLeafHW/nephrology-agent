# Nephrology Knowledge Assistant

A standalone Streamlit RAG application for general nephrology education. It extracts text from authorized PDFs, embeds chunks with Gemini Embeddings, stores the vectors and source metadata locally in ChromaDB, retrieves relevant excerpts, and asks Gemini Flash-Lite to answer with numbered citations.

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

Enter your key from [Google AI Studio](https://aistudio.google.com/apikey) in the app's masked sidebar field. It is used for that app session and is not written to disk. Keep `.env` private; it is excluded by `.gitignore`. Never paste API keys into chat, screenshots, or source code. If a key is exposed, revoke it and create a replacement.

Run the app:

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Upload PDFs from reputable nephrology sources in the sidebar, confirm that you have permission to index them, and optionally add the HTTPS source URL. The app preserves file name, page number, and source URL with each indexed chunk. Each browser session gets a separate in-memory knowledge base; indexed documents are not written to disk or shared across visitors. Session data may remain in server memory until the app process restarts. It does not ship a preloaded PDF corpus: choose and index current sources you are allowed to use. [KDIGO guidelines](https://kdigo.org/guidelines/) and [NIDDK kidney health information](https://www.niddk.nih.gov/health-information/kidney-disease) are useful starting points; check each document's license and update date before indexing.

Text extraction does not include scanned-image OCR. OCR scanned PDFs before uploading. Each file is limited to 20 MB. Indexed text and vectors are held in memory for the session; uploaded PDF files are not retained as files.

## Privacy and medical safety

This is an educational prototype, not a medical device or clinical decision-support system. It does not diagnose, interpret individual patient results, or recommend treatment changes. A clinician should be consulted for personal medical decisions; urgent symptoms need timely professional care.

The vector store and chat history are temporary and private to each browser session, but embedding requests send PDF text to Google's API, and question-answer requests send the question and retrieved excerpts. Your Gemini API key is sent to the app server to make those requests and is not saved by this app. Google's free-tier terms state that submitted content may be used to improve its products. Do not upload protected health information or enter identifiable patient data. Review Google's current [API terms](https://ai.google.dev/gemini-api/terms) and [pricing/data-use details](https://ai.google.dev/gemini-api/docs/pricing) before use.

## Public hosting

The app can be deployed from this repository to [Streamlit Community Cloud](https://share.streamlit.io/). Select branch `main` and `app.py` as the entry point; no hosted API key is needed because each visitor supplies their own. The app is public and does not require sign-in, so anyone with the link can use it. Free-tier availability and limits may change. Session data is not persisted to disk; its in-memory collection may remain until the app process restarts.

Answers are grounded in the indexed excerpts and include source references, but retrieval can miss relevant evidence and generated answers can be wrong or incomplete. Verify important claims in the cited source and consult a qualified professional.

## Tests

```powershell
python -m unittest discover -s tests
```
