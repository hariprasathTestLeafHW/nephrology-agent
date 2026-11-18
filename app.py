import os
from pathlib import Path

import streamlit as st
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
from pypdf.errors import PdfReadError
from streamlit.errors import StreamlitSecretNotFoundError

from nephrology_agent.rag import (
    answer_question,
    create_shared_persistent_collection,
    index_preloaded_pdfs,
)

APP_DIRECTORY = Path(__file__).resolve().parent
PRELOADED_PDF_DIRECTORY = APP_DIRECTORY / "data" / "knowledge_base"
VECTOR_STORE_DIRECTORY = APP_DIRECTORY / "data" / "chroma"
ASSISTANT_ILLUSTRATION = APP_DIRECTORY / "assets" / "nephrology_assistant.svg"
PRELOADED_SOURCE_LABELS = {
    "nephrology_comprehensive_symptom_diagnosis_rag_kb.pdf": "Symptom and diagnosis reference",
    "nephrology_diagnosis_agent_rag_starter.pdf": "Nephrology assistant starter guide",
}
SUGGESTED_QUESTIONS = (
    "What kidney symptoms need urgent attention?",
    "How are abnormal kidney tests generally evaluated?",
    "What can cause blood or protein in urine?",
)

load_dotenv(APP_DIRECTORY / ".env")

st.set_page_config(
    page_title="Nephrology Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
)

st.html(APP_DIRECTORY / "assets" / "glass.css")

with st.container(key="hero-card"):
    intro_column, illustration_column = st.columns(
        [1.3, 0.7],
        vertical_alignment="center",
    )
    with intro_column:
        st.badge(
            "Nephrology education",
            icon=":material/medical_services:",
            color="green",
        )
        st.title("Nephrology Knowledge Assistant")
        st.markdown(
            "### Kidney health, explained with care."
            "\n\nAsk general questions and explore clear answers grounded in "
            "curated nephrology references."
        )
        st.caption("Powered by Google Gemini LLM")
    with illustration_column:
        st.image(
            ASSISTANT_ILLUSTRATION,
            width="stretch",
            alt="Illustration of a nephrology assistant beside a kidney health diagram.",
        )

with st.container(key="safety-card"):
    st.markdown(
        "**Educational use only — not a substitute for professional medical care.** "
        "Do not enter patient-identifiable or other sensitive health information."
    )
    st.caption(
        "Questions, retrieved excerpts, and PDF text for embedding are sent to Google's "
        "Gemini API. Google's free-tier terms may allow submitted content to be used "
        "to improve its products."
    )

st.markdown(
    "[KDIGO guidelines](https://kdigo.org/guidelines/) · "
    "[NIDDK kidney health information](https://www.niddk.nih.gov/health-information/kidney-disease)"
)

model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
embedding_model = os.environ.get(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
).strip()


@st.cache_resource(max_entries=1)
def load_preloaded_collection(
    _api_key: str,
    configured_embedding_model: str,
) -> tuple[ClientAPI, Collection]:
    chroma_client, shared_collection = create_shared_persistent_collection(
        VECTOR_STORE_DIRECTORY
    )
    index_preloaded_pdfs(
        client=genai.Client(api_key=_api_key),
        collection=shared_collection,
        pdf_directory=PRELOADED_PDF_DIRECTORY,
        embedding_model=configured_embedding_model,
    )
    return chroma_client, shared_collection


try:
    configured_api_key = st.secrets["GEMINI_API_KEY"].strip()
except (KeyError, StreamlitSecretNotFoundError):
    configured_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

with st.sidebar:
    st.header("Knowledge base")
    if configured_api_key:
        st.caption("RAG content")
        for pdf_path in sorted(PRELOADED_PDF_DIRECTORY.glob("*.pdf")):
            st.write(
                PRELOADED_SOURCE_LABELS.get(
                    pdf_path.name,
                    pdf_path.stem.replace("_", " ").title(),
                )
            )
    st.caption("The server caches the vector index; rebuilding it uses Gemini quota.")

collection: Collection | None = None
document_count = 0
if configured_api_key:
    try:
        with st.spinner("Loading the preloaded nephrology library..."):
            _, collection = load_preloaded_collection(
                configured_api_key,
                embedding_model,
            )
    except (OSError, PdfReadError, ValueError, RuntimeError) as error:
        st.error(f"Could not load the preloaded nephrology library: {error}")
        st.stop()
    except APIError as error:
        st.error(f"Gemini API request failed while loading the preloaded library: {error}")
        st.stop()
    document_count = collection.count()

st.sidebar.metric("Indexed text chunks", document_count)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = (
        ":material/medical_services:"
        if message["role"] == "assistant"
        else None
    )
    with st.chat_message(message["role"], avatar=avatar):
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

suggested_question = None
if configured_api_key and document_count and not st.session_state.messages:
    st.markdown("#### Start with a question")
    st.caption("Choose an example or type your own below.")
    suggested_question = st.pills(
        "Suggested questions",
        SUGGESTED_QUESTIONS,
        selection_mode="single",
        label_visibility="collapsed",
        key="suggested_question",
    )

question = st.chat_input(
    "Ask a general nephrology question",
    disabled=not configured_api_key or document_count == 0,
)
question = question or suggested_question

if question and collection is not None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/medical_services:"):
        with st.spinner("Searching indexed sources and preparing an answer..."):
            try:
                answer, sources = answer_question(
                    client=genai.Client(api_key=configured_api_key),
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
