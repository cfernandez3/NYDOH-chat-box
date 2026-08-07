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
    initial_sidebar_state="expanded",
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

st.markdown(
    """
    <style>
    .stApp {background:#f4f7f9;}
    .main .block-container {max-width:1180px;padding-top:1.2rem;padding-bottom:2rem;}
    .hero {background:linear-gradient(135deg,#0f3f56 0%,#1c6b79 100%);color:white;
           padding:1.6rem 1.9rem;border-radius:18px;margin-bottom:1.25rem;
           box-shadow:0 8px 24px rgba(15,63,86,.16);}
    .hero h1 {margin:0 0 .35rem 0;font-size:2rem;line-height:1.15;}
    .hero p {margin:0;opacity:.92;font-size:1rem;}
    .answer-card,.panel {background:white;border:1px solid #dce5ea;border-radius:16px;
                         padding:1.2rem 1.35rem;box-shadow:0 4px 16px rgba(15,63,86,.06);}
    .answer-card {min-height:220px;}
    .source-card {background:#f9fbfc;border:1px solid #dce5ea;border-left:5px solid #2c7a62;
                  border-radius:12px;padding:.85rem .95rem;margin-bottom:.7rem;}
    .source-title {color:#123f56;font-weight:750;margin-bottom:.35rem;}
    .source-text {color:#465a67;font-size:.92rem;line-height:1.45;}
    .eyebrow {font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;
              color:#5d7380;font-weight:750;margin-bottom:.4rem;}
    .empty-state {color:#687c88;padding:2.2rem .5rem;text-align:center;}
    section[data-testid="stSidebar"] {background:#eaf1f4;border-right:1px solid #d4e0e5;}
    div[data-testid="stTextArea"] textarea {border-radius:13px;border:1px solid #c9d7de;font-size:1rem;}
    div.stButton>button {border-radius:11px;font-weight:700;min-height:2.8rem;}
    .small-note {color:#627681;font-size:.85rem;line-height:1.4;}
    </style>
    """,
    unsafe_allow_html=True,
)

def get_openai_api_key():
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY is not configured in Streamlit Secrets.")
        st.stop()
    return key

def download_pdf(url, destination):
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)

def normalize_standard(value):
    return re.sub(r"\s+", " ", value).strip().upper()

def extract_standard(text):
    match = re.search(r"\b[A-Z]{1,10}\s*S\d+(?:\.\d+)?\b", text, re.IGNORECASE)
    return normalize_standard(match.group(0)) if match else "Not identified"

@st.cache_resource(show_spinner="Loading NYDOH standards and preparing the search index...")
def initialize_rag(api_key):
    general_path = DATA_DIR / "general.pdf"
    histo_path = DATA_DIR / "histocompatibility.pdf"
    download_pdf(GENERAL_URL, general_path)
    download_pdf(HISTO_URL, histo_path)

    all_pages = []
    for pdf_path, source_name in [
        (general_path, "NYDOH General"),
        (histo_path, "NYDOH Histocompatibility"),
    ]:
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.metadata["source"] = source_name
            page.metadata["standard"] = extract_standard(page.page_content)
        all_pages.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_pages)
    for chunk in chunks:
        standard = extract_standard(chunk.page_content)
        if standard != "Not identified":
            chunk.metadata["standard"] = standard

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="nydoh_standards",
    )
    return vector_db, OpenAI(api_key=api_key), len(all_pages), len(chunks)

def asks_for_compliance_ideas(query):
    """
    Detect whether the user is asking how to implement or comply with a standard.
    This intentionally uses a conservative phrase-based rule so normal standards
    questions remain document-only.
    """
    normalized = re.sub(r"\s+", " ", query.lower()).strip()

    compliance_phrases = [
        "how do we comply",
        "how can we comply",
        "how should we comply",
        "how to comply",
        "ways to comply",
        "steps to comply",
        "help us comply",
        "implement this standard",
        "implement the standard",
        "how do we implement",
        "how can we implement",
        "how should we implement",
        "implementation ideas",
        "compliance ideas",
        "best practices for",
        "practical steps",
        "what should the laboratory do",
        "what can the laboratory do",
    ]

    return any(phrase in normalized for phrase in compliance_phrases)


def get_web_compliance_ideas(query, client):
    """
    Search the public web for practical implementation ideas.

    This is intentionally separate from the NYDOH-grounded answer so users can
    distinguish regulatory requirements from outside suggestions.
    """
    web_prompt = f"""
The user is asking for practical ideas for complying with an NYDOH clinical
laboratory standard.

USER QUESTION:
{query}

Search the public web for practical, non-vendor-specific implementation ideas.
Prioritize authoritative sources such as government agencies, accreditation
organizations, professional laboratory organizations, universities, and
recognized quality-management sources.

Do not present internet material as an NYDOH requirement.
Do not provide legal advice.
Do not include or request patient-identifiable information.

Return:
1. A short heading: "Proposed implementation ideas"
2. Four to seven concise, practical ideas.
3. A short "Web sources consulted" list with source names and URLs.
4. End with this exact disclaimer:

"These are proposed solutions found on the internet, but they must be confirmed / verified by the Lab Director."
"""

    response = client.responses.create(
        model="gpt-5-mini",
        tools=[{"type": "web_search"}],
        input=web_prompt,
    )

    return response.output_text.strip()


def answer_question(query, vector_db, client, k=2):
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
- Use concise headings or bullets when helpful.

RULES:
- Use ONLY the provided context.
- Do NOT hallucinate or invent information.
- Do NOT use prior knowledge.
- Do NOT reference external sources not in the context.
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

    sources, seen = [], set()
    for doc in docs:
        metadata = doc.metadata
        standard = metadata.get("standard") or extract_standard(doc.page_content)
        page = metadata.get("page_label")
        if page is None:
            raw_page = metadata.get("page")
            page = str(raw_page + 1) if isinstance(raw_page, int) else "Unknown"

        item = {
            "standard": standard,
            "source": metadata.get("source", "Unknown"),
            "page": str(page),
            "context": re.sub(r"\s+", " ", doc.page_content).strip(),
        }
        key = (item["standard"], item["source"], item["page"])
        if key not in seen:
            seen.add(key)
            sources.append(item)

    grounded_answer = response.choices[0].message.content.strip()
    web_ideas = None

    if asks_for_compliance_ideas(query):
        try:
            web_ideas = get_web_compliance_ideas(query, client)
        except Exception as exc:
            web_ideas = (
                "### Proposed implementation ideas\n\n"
                "A web search could not be completed at this time. "
                "The NYDOH-grounded answer above is still available.\n\n"
                "**Disclaimer:** These are proposed solutions found on the internet, "
                "but they must be confirmed / verified by the Lab Director.\n\n"
                f"_Web-search error: {exc}_"
            )

    return {
        "answer": grounded_answer,
        "sources": sources,
        "web_ideas": web_ideas,
    }

if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "history" not in st.session_state:
    st.session_state.history = []

api_key = get_openai_api_key()
vector_db, client, page_count, chunk_count = initialize_rag(api_key)

with st.sidebar:
    st.markdown("## NYDOH Assistant")
    st.caption("Clinical Laboratory Standards")

    retrieval_k = 2
    st.markdown("**Retrieved passages:** 2")
    st.caption("This setting is fixed to preserve consistent retrieval behavior.")

    st.markdown("---")
    st.markdown(f"**Documents loaded:** {page_count} pages")
    st.markdown(f"**Indexed sections:** {chunk_count}")
    st.markdown("---")
    st.markdown("### Recent questions")

    if not st.session_state.history:
        st.caption("No questions yet.")
    else:
        for index, item in enumerate(reversed(st.session_state.history[-8:])):
            label = item["question"]
            if len(label) > 42:
                label = label[:42] + "…"
            if st.button(label, key=f"history_{index}", use_container_width=True):
                st.session_state.current_question = item["question"]
                st.session_state.current_result = item["result"]
                st.rerun()

    if st.session_state.history and st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_result = None
        st.session_state.current_question = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="small-note">'
        '<strong>NYDOH answers:</strong> based only on the indexed standards.<br><br>'
        '<strong>Compliance ideas:</strong> a public web search is used only when the '
        'question asks how to comply or implement a standard. Do not enter patient-identifiable information.<br><br>'
        'Verify critical compliance decisions against the official documents.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
        <h1>🧪 NYDOH Clinical Laboratory Standards Assistant</h1>
        <p>Ask questions about the General Systems and Histocompatibility standards.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Ask a question</div>', unsafe_allow_html=True)
question = st.text_area(
    "Question",
    value=st.session_state.current_question,
    placeholder="Example: What does NYDOH require for crossmatching in histocompatibility?",
    height=105,
    label_visibility="collapsed",
)

c1, c2, _ = st.columns([1, 1, 5])
with c1:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with c2:
    new_clicked = st.button("New question", use_container_width=True)

if new_clicked:
    st.session_state.current_question = ""
    st.session_state.current_result = None
    st.rerun()

if ask_clicked:
    cleaned = question.strip()
    if not cleaned:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Reviewing the NYDOH standards..."):
            result = answer_question(cleaned, vector_db, client, retrieval_k)
        st.session_state.current_question = cleaned
        st.session_state.current_result = result
        st.session_state.history.append({"question": cleaned, "result": result})
        st.rerun()

st.markdown("---")
left, right = st.columns([1.7, 1])

with left:
    st.markdown('<div class="eyebrow">Current answer</div>', unsafe_allow_html=True)
    if st.session_state.current_result is None:
        st.markdown(
            '<div class="answer-card"><div class="empty-state">Enter a question above to begin.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.current_result["answer"])

        web_ideas = st.session_state.current_result.get("web_ideas")
        if web_ideas:
            st.markdown("---")
            st.markdown(web_ideas)

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="eyebrow">Supporting standards</div>', unsafe_allow_html=True)
    if st.session_state.current_result is None:
        st.markdown(
            '<div class="panel"><div class="empty-state">Sources will appear here.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        sources = st.session_state.current_result["sources"]
        if not sources:
            st.info("No supporting passages were retrieved.")
        else:
            for source in sources:
                excerpt = source["context"][:520]
                if len(source["context"]) > 520:
                    excerpt += "…"
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">{source['standard']} | Page {source['page']}</div>
                        <div class="source-text">{excerpt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
