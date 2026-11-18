import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from pypdf.errors import PdfReadError
from streamlit.errors import StreamlitSecretNotFoundError

from nephrology_agent.rag import (
    MAX_PDF_BYTES,
    answer_question,
    create_ephemeral_collection,
    index_pdf,
)

APP_DIRECTORY = Path(__file__).resolve().parent

load_dotenv(APP_DIRECTORY / ".env")

st.set_page_config(
    page_title="Nephrology Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("Nephrology Knowledge Assistant")
st.caption("An educational assistant that answers from your indexed nephrology PDFs.")
st.warning(
    "Educational use only — not a substitute for professional medical care. "
    "Do not enter patient-identifiable or other sensitive health information. "
    "This app sends questions, retrieved excerpts, and PDF text embeddings to Google's Gemini API. "
    "Google's free-tier terms may allow submitted content to be used to improve its products."
)
st.markdown(
    "[Google AI Studio API keys](https://aistudio.google.com/apikey) · "
    "[KDIGO guidelines](https://kdigo.org/guidelines/) · "
    "[NIDDK kidney health information](https://www.niddk.nih.gov/health-information/kidney-disease)"
)

model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
embedding_model = os.environ.get(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
).strip()

try:
    configured_api_key = st.secrets["GEMINI_API_KEY"].strip()
except (KeyError, StreamlitSecretNotFoundError):
    configured_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

with st.sidebar:
    st.header("Knowledge base")
    if configured_api_key:
        api_key = configured_api_key
        st.warning(
            "The app's Gemini key is active. Requests from any visitor use the "
            "app owner's Gemini quota."
        )
    else:
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            help="Used for this session's Gemini requests and not saved by the app.",
        ).strip()
    st.write(
        "Add PDFs you are authorized to use. Indexed text and vectors are private to "
        "this browser session."
    )
    st.caption("Check each document's license and update date before indexing.")
    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    source_url = st.text_input("Source URL (optional)", placeholder="https://...")
    rights_confirmed = st.checkbox(
        "I have permission to index these documents and preserve their source attribution."
    )
    add_documents = st.button(
        "Index selected PDFs",
        disabled=not (uploaded_files and rights_confirmed and api_key),
        use_container_width=True,
    )

    if not configured_api_key and not api_key:
        st.info("Enter your Gemini API key to enable indexing and chat.")

if "chroma_client" not in st.session_state:
    st.session_state["chroma_client"], st.session_state["collection"] = (
        create_ephemeral_collection()
    )
collection = st.session_state["collection"]
document_count = collection.count()
chunk_metric = st.sidebar.empty()
chunk_metric.metric("Indexed text chunks", document_count)

if add_documents and uploaded_files:
    ai_client = genai.Client(api_key=api_key)
    for uploaded_file in uploaded_files:
        if uploaded_file.size > MAX_PDF_BYTES:
            st.error(f"{uploaded_file.name} exceeds the 20 MB per-PDF limit.")
            continue
        try:
            chunk_count, already_indexed = index_pdf(
                client=ai_client,
                collection=collection,
                filename=uploaded_file.name,
                content=uploaded_file.getvalue(),
                embedding_model=embedding_model,
                source_url=source_url.strip(),
            )
        except (PdfReadError, ValueError) as error:
            st.error(f"Could not index {uploaded_file.name}: {error}")
        except APIError as error:
            st.error(f"Gemini API request failed while indexing {uploaded_file.name}: {error}")
            break
        else:
            if already_indexed:
                st.info(f"{uploaded_file.name} is already indexed.")
            else:
                st.success(f"Indexed {chunk_count} text chunks from {uploaded_file.name}.")

    document_count = collection.count()
    chunk_metric.metric("Indexed text chunks", document_count)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.write(
                        f"[{source['number']}] {source['filename']} — page {source['page']}"
                    )
                    if source["source_url"]:
                        st.link_button(
                            f"Open source [{source['number']}]",
                            source["source_url"],
                        )

question = st.chat_input(
    "Ask a general nephrology question",
    disabled=not api_key or document_count == 0,
)

if not api_key:
    st.info("Add a Gemini API key to start. Get one from Google AI Studio.")
elif document_count == 0:
    st.info("Index at least one authorized PDF in the sidebar before asking a question.")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching indexed sources and preparing an answer..."):
            try:
                answer, sources = answer_question(
                    client=genai.Client(api_key=api_key),
                    collection=collection,
                    question=question,
                    model=model,
                    embedding_model=embedding_model,
                )
            except ValueError as error:
                st.error(str(error))
                st.stop()
            except APIError as error:
                st.error(f"Gemini API request failed: {error}")
                st.stop()
            except RuntimeError as error:
                st.error(str(error))
                st.stop()

        st.markdown(answer)
        with st.expander("Retrieved sources"):
            for source in sources:
                st.write(f"[{source['number']}] {source['filename']} — page {source['page']}")
                if source["source_url"]:
                    st.link_button(
                        f"Open source [{source['number']}]",
                        source["source_url"],
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
