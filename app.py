import os
import re
from pathlib import Path

import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

st.set_page_config(
    page_title="NYDOH Standards Assistant",
    page_icon="🧪",
    layout="wide",
)

GENERAL_URL = (
    "https://raw.githubusercontent.com/cfernandez3/NYDOH-chat-box/main/"
    "General%20Systems%20Standards%20Effective%20February%202026%20v2.1.pdf"
)
HISTO_URL = (
    "https://raw.githubusercontent.com/cfernandez3/NYDOH-chat-box/main/"
    "Histocompatibility%20-%20Effective%20February%202026.pdf"
)

DATA_DIR = Path("data")
VECTOR_DIR = Path("vectorstore_chroma")


def get_openai_api_key() -> str:
    """Read the API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = os.getenv("OPENAI_API_KEY")

    if not key:
        st.error(
            "OPENAI_API_KEY is not configured. Add it to Streamlit Secrets "
            "or to .streamlit/secrets.toml."
        )
        st.stop()

    return key


def download_pdf(url: str, destination: Path) -> None:
    """Download a PDF only when it is not already present."""
    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def normalize_standard(value: str) -> str:
    """Normalize PDF whitespace, e.g. 'HC\\nS5' -> 'HC S5'."""
    return re.sub(r"\s+", " ", value).strip().upper()


def extract_standard(text: str) -> str:
    """Extract codes such as HC S1, QMS S2, or GEN S3.1."""
    pattern = r"\b[A-Z]{1,10}\s*S\d+(?:\.\d+)?\b"
    match = re.search(pattern, text, re.IGNORECASE)
    return normalize_standard(match.group(0)) if match else "Not identified"


@st.cache_resource(show_spinner="Loading NYDOH standards and building the search index...")
def initialize_rag(api_key: str):
    """
    Download, load, split, embed, and index the NYDOH standards.

    Streamlit caches the returned resources, so this expensive setup is not
    repeated on every user interaction.
    """
    general_path = DATA_DIR / "general.pdf"
    histo_path = DATA_DIR / "histocompatibility.pdf"

    download_pdf(GENERAL_URL, general_path)
    download_pdf(HISTO_URL, histo_path)

    document_specs = [
        (general_path, "NYDOH General"),
        (histo_path, "NYDOH Histocompatibility"),
    ]

    all_pages = []

    for pdf_path, source_name in document_specs:
        pages = PyPDFLoader(str(pdf_path)).load()

        for page in pages:
            page.metadata["source"] = source_name

            # Save the first standard found on the full page as metadata.
            standard = extract_standard(page.page_content)
            page.metadata["standard"] = standard

        all_pages.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(all_pages)

    # A chunk may contain a more specific standard than the first one on its page.
    for chunk in chunks:
        chunk_standard = extract_standard(chunk.page_content)
        if chunk_standard != "Not identified":
            chunk.metadata["standard"] = chunk_standard

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    # Build the vector store in memory. Streamlit caches it across reruns.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="nydoh_standards",
    )

    client = OpenAI(api_key=api_key)
    return vector_db, client, len(all_pages), len(chunks)


def answer_question(
    query: str,
    vector_db: Chroma,
    client: OpenAI,
    k: int = 2,
) -> dict:
    """Retrieve relevant standards and generate a grounded answer."""
    docs = vector_db.similarity_search(query, k=k)

    if not docs:
        return {
            "answer": "I do not see a clear answer in the provided NYDOH materials.",
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
CONTEXT:
{context}

INSTRUCTIONS:
- You are a compliance assistant for NYDOH Clinical Laboratory Standards of Practice.
- Help laboratory professionals understand regulatory requirements.
- Answer clearly and professionally.

RULES:
- Use ONLY the provided context.
- Do NOT hallucinate or invent information.
- Do NOT use prior knowledge.
- Do NOT reference external laws, websites, or sources not in the context.
- If the answer is not in the context, say:
  "I do not see a clear answer in the provided NYDOH materials."
- When possible, identify the relevant standard.

QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )

    answer = response.choices[0].message.content.strip()

    sources = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata
        standard = metadata.get("standard") or extract_standard(doc.page_content)
        page = metadata.get("page_label")

        # PyPDFLoader page is zero-based. Use it only if page_label is absent.
        if page is None:
            raw_page = metadata.get("page")
            page = str(raw_page + 1) if isinstance(raw_page, int) else "Unknown"

        item = {
            "standard": standard,
            "source": metadata.get("source", "Unknown"),
            "page": str(page),
            "context": doc.page_content,
        }

        key = (item["standard"], item["source"], item["page"])
        if key not in seen:
            seen.add(key)
            sources.append(item)

    return {"answer": answer, "sources": sources}


api_key = get_openai_api_key()
vector_db, client, page_count, chunk_count = initialize_rag(api_key)

st.title("NYDOH Clinical Laboratory Standards Assistant")
st.caption(
    "Ask questions about the General Systems and Histocompatibility standards. "
    "Answers are generated only from the indexed NYDOH documents."
)

with st.sidebar:
    st.header("About")
    st.write(f"Loaded pages: **{page_count}**")
    st.write(f"Indexed chunks: **{chunk_count}**")
    retrieval_k = st.select_slider(
        "Number of retrieved passages",
        options=[1, 2, 3, 4],
        value=2,
        help="Two is a practical starting point. More passages may improve recall but reduce precision.",
    )
    st.warning(
        "This tool supports review of the supplied standards. "
        "Always verify critical compliance decisions against the official document."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    label = (
                        f"{source['standard']} | "
                        f"{source['source']} | Page {source['page']}"
                    )
                    st.markdown(f"**{label}**")
                    st.caption(source["context"][:700] + ("…" if len(source["context"]) > 700 else ""))

question = st.chat_input("Ask a question about the NYDOH standards")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing the NYDOH standards..."):
            try:
                result = answer_question(
                    query=question,
                    vector_db=vector_db,
                    client=client,
                    k=retrieval_k,
                )
            except Exception as exc:
                st.error(f"The question could not be processed: {exc}")
                st.stop()

        st.markdown(result["answer"])

        with st.expander("Sources", expanded=True):
            if not result["sources"]:
                st.write("No supporting passages were retrieved.")
            else:
                for source in result["sources"]:
                    label = (
                        f"{source['standard']} | "
                        f"{source['source']} | Page {source['page']}"
                    )
                    st.markdown(f"**{label}**")
                    st.caption(
                        source["context"][:700]
                        + ("…" if len(source["context"]) > 700 else "")
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
