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
    
    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 2px solid #d9534f;
        background: #fff1f0;
    }
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


def extract_standard_title(text, standard_code):
    """
    Extract a short title appearing near the standard code.
    Example: HC S5 -> Crossmatching
    """
    if not standard_code or standard_code == "Not identified":
        return "Title not identified"

    compact = re.sub(r"\s+", " ", text).strip()

    patterns = [
        rf"{re.escape(standard_code)}\s*[:\-]\s*([^.;]{{3,120}})",
        rf"{re.escape(standard_code)}\)?\s*([^.;]{{3,120}})",
    ]

    for pattern in patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            title = match.group(1).strip(" :-")
            if len(title) > 90:
                title = title[:90].rstrip() + "…"
            return title

    return "Title not identified"


def split_paragraphs(text):
    paragraphs = [
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(r"\n\s*\n", text)
        if item.strip()
    ]

    if len(paragraphs) <= 1:
        paragraphs = [
            re.sub(r"\s+", " ", item).strip()
            for item in re.split(r"(?<=[.;:])\s+(?=[A-Z0-9])", text)
            if item.strip()
        ]

    return paragraphs


def best_supporting_paragraph(query, text):
    """
    Select the most query-relevant paragraph from a retrieved chunk.
    This is deterministic and does not generate new wording.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return re.sub(r"\s+", " ", text).strip()

    stopwords = {
        "what", "when", "where", "which", "with", "that", "this", "from",
        "have", "does", "must", "shall", "about", "into", "your", "their",
        "how", "are", "the", "and", "for", "can", "should", "would",
    }

    query_terms = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if len(token) > 2 and token not in stopwords
    }

    def score(paragraph):
        lower = paragraph.lower()
        overlap = sum(1 for term in query_terms if term in lower)
        standard_bonus = 2 if re.search(r"\b[A-Z]{1,10}\s*S\d+", paragraph) else 0
        return overlap + standard_bonus

    return max(paragraphs, key=score)


def highlight_query_terms(text, query):
    terms = sorted(
        {
            token for token in re.findall(r"[a-zA-Z0-9]+", query)
            if len(token) > 3
        },
        key=len,
        reverse=True,
    )

    highlighted = text
    for term in terms[:10]:
        highlighted = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            r"<mark>\1</mark>",
            highlighted,
        )

    return highlighted


def standard_search_query(query):
    """
    Detect direct searches such as 'HC S5' or 'show me QMS S2'.
    """
    match = re.search(r"\b[A-Z]{1,10}\s*S\d+(?:\.\d+)?\b", query, re.IGNORECASE)
    return normalize_standard(match.group(0)) if match else None


def find_chunks_by_standard(chunks, standard_code):
    exact = []

    for chunk in chunks:
        metadata_code = normalize_standard(
            str(chunk.metadata.get("standard", ""))
        )

        content_code = extract_standard(chunk.page_content)

        if standard_code in {metadata_code, content_code}:
            exact.append(chunk)

    return exact


def build_compliance_checklist(question, answer, source_text, client):
    """
    Create a checklist using only the grounded answer and NYDOH source text.
    """
    prompt = f"""
Create a concise compliance checklist using ONLY the NYDOH material below.

QUESTION:
{question}

GROUNDED ANSWER:
{answer}

NYDOH SOURCE TEXT:
{source_text}

RULES:
- Return 4 to 8 checklist items.
- Start every item with "- [ ]".
- Do not invent requirements.
- If the source does not support a checklist item, omit it.
- Use operational wording such as "Maintain", "Document", "Review", or "Verify".
- Do not add internet-derived implementation ideas.

CHECKLIST:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=450,
    )

    return response.choices[0].message.content.strip()


def explain_standard_connections(question, related_sources, client):
    """
    Explain how multiple retrieved standards relate, using only their text.
    """
    if len(related_sources) < 2:
        return None

    context = "\n\n".join(
        f"{item['standard']} — {item['title']}:\n{item['context']}"
        for item in related_sources[:4]
    )

    prompt = f"""
Using ONLY the NYDOH excerpts below, explain briefly how the standards connect
to the user's question.

USER QUESTION:
{question}

NYDOH EXCERPTS:
{context}

RULES:
- Use 2 to 4 bullets.
- Name each standard explicitly.
- Do not invent relationships not supported by the excerpts.
- Distinguish primary requirements from supporting or related requirements.

CONNECTIONS:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()



def standard_sort_key(code):
    match = re.match(
        r"^([A-Z]{1,10})\s+S(\d+)(?:\.(\d+))?$",
        normalize_standard(code),
    )
    if not match:
        return (normalize_standard(code), 9999, 9999)

    prefix, major, minor = match.groups()
    return (prefix, int(major), int(minor or 0))


def build_standard_catalog(chunks):
    """
    Build a unique catalog of all identifiable standards in the indexed PDFs.
    """
    catalog = {}
    pattern = re.compile(
        r"\b([A-Z]{1,10})\s*S(\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )

    for chunk in chunks:
        compact = re.sub(r"\s+", " ", chunk.page_content).strip()

        for match in pattern.finditer(compact):
            code = normalize_standard(
                f"{match.group(1)} S{match.group(2)}"
            )

            metadata = chunk.metadata
            source_name = metadata.get("source", "Unknown")

            page = metadata.get("page_label")
            if page is None:
                raw_page = metadata.get("page")
                page = str(raw_page + 1) if isinstance(raw_page, int) else "Unknown"

            pdf_url = GENERAL_URL if source_name == "NYDOH General" else HISTO_URL
            title = extract_standard_title(compact, code)

            excerpt_start = max(0, match.start() - 40)
            excerpt_end = min(len(compact), match.end() + 520)
            excerpt = compact[excerpt_start:excerpt_end].strip()

            candidate = {
                "standard": code,
                "prefix": code.split()[0],
                "title": title,
                "source": source_name,
                "page": str(page),
                "pdf_link": f"{pdf_url}#page={page}",
                "excerpt": excerpt,
            }

            existing = catalog.get(code)
            if existing is None:
                catalog[code] = candidate
            elif (
                existing["title"] == "Title not identified"
                and title != "Title not identified"
            ):
                catalog[code] = candidate

    return dict(
        sorted(
            catalog.items(),
            key=lambda item: standard_sort_key(item[0]),
        )
    )


def summarize_catalog_entry(entry):
    text = entry["excerpt"]
    text = re.sub(
        rf"(?i).*?{re.escape(entry['standard'])}\s*[:\-]?\s*",
        "",
        text,
        count=1,
    ).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:2]).strip()

    if len(summary) > 320:
        summary = summary[:320].rstrip() + "…"

    return summary or "Summary not available from the indexed excerpt."


def detect_index_request(query, prefixes):
    normalized = re.sub(r"\s+", " ", query.lower()).strip()

    full_index_phrases = [
        "show the index",
        "show me the index",
        "standards index",
        "list all standards",
        "show all standards",
        "table of contents",
        "list the sections",
        "show the sections",
    ]

    for prefix in prefixes:
        prefix_lower = prefix.lower()

        if re.search(rf"\b{re.escape(prefix_lower)}\b", normalized):
            intent_words = [
                "all",
                "section",
                "standards",
                "list",
                "index",
                "show",
                "summary",
                "summarize",
            ]

            if any(word in normalized for word in intent_words):
                return {"type": "section", "prefix": prefix}

    if any(phrase in normalized for phrase in full_index_phrases):
        return {"type": "full", "prefix": None}

    return None


def format_standard_index(catalog, prefix=None):
    entries = list(catalog.values())

    if prefix:
        entries = [
            item for item in entries
            if item["prefix"] == prefix
        ]

    if not entries:
        return (
            f"No indexed standards were found for section **{prefix}**."
            if prefix
            else "No indexed standards were found."
        )

    if prefix:
        lines = [
            f"## {prefix} Standards",
            "",
            f"**{len(entries)} standards found.**",
            "",
        ]
    else:
        prefixes = sorted(
            {item["prefix"] for item in entries}
        )
        lines = [
            "## NYDOH Standards Index",
            "",
            (
                f"**{len(entries)} standards found across "
                f"{len(prefixes)} sections.**"
            ),
            "",
        ]

    grouped = {}
    for item in entries:
        grouped.setdefault(item["prefix"], []).append(item)

    for section in sorted(grouped):
        if not prefix:
            lines.extend([f"### {section}", ""])

        for item in sorted(
            grouped[section],
            key=lambda row: standard_sort_key(row["standard"]),
        ):
            summary = summarize_catalog_entry(item)

            lines.extend(
                [
                    (
                        f"**[{item['standard']} — {item['title']}]"
                        f"({item['pdf_link']})**"
                    ),
                    f"Page {item['page']} · {item['source']}",
                    "",
                    summary,
                    "",
                ]
            )

    return "\n".join(lines)


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
    standard_catalog = build_standard_catalog(chunks)

    return (
        vector_db,
        OpenAI(api_key=api_key),
        len(all_pages),
        len(chunks),
        chunks,
        standard_catalog,
    )

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


def expand_query_for_retrieval(query):
    """
    Add standards-oriented concepts to the retrieval query without changing
    the user's displayed question.

    This helps connect common healthcare terminology (for example HIPAA or PHI)
    to wording more likely to appear in the NYDOH standards, such as
    confidentiality, privacy, authorized access, disclosure, and patient data.
    """
    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    expansions = []

    concept_map = {
        "hipaa": (
            "patient information confidentiality privacy protected health information "
            "authorized access unauthorized access disclosure release of information "
            "security of patient data personnel confidentiality training"
        ),
        "phi": (
            "protected health information patient information confidentiality privacy "
            "authorized access disclosure release of information"
        ),
        "patient data": (
            "patient information confidentiality privacy authorized access disclosure "
            "security records"
        ),
        "patient information": (
            "confidentiality privacy authorized access disclosure release of information "
            "security records"
        ),
        "confidentiality": (
            "patient information privacy authorized access unauthorized disclosure "
            "release of information personnel training"
        ),
        "privacy": (
            "patient information confidentiality authorized access disclosure "
            "release of information"
        ),
        "medical records": (
            "patient records confidentiality privacy access disclosure retention security"
        ),
        "cybersecurity": (
            "information security access control confidentiality integrity availability "
            "electronic records"
        ),
        "data security": (
            "information security access control confidentiality integrity availability "
            "electronic records"
        ),
        "access control": (
            "authorized access unique user identification permissions patient information "
            "confidentiality"
        ),
    }

    for phrase, expansion in concept_map.items():
        if phrase in normalized:
            expansions.append(expansion)

    if not expansions:
        return query

    return f"{query}\nRelated NYDOH concepts: {' '.join(expansions)}"


def get_concept_note(query):
    """
    Provide a short, careful bridge when the user uses a common external term
    that may not appear verbatim in the NYDOH standards.
    """
    normalized = query.lower()

    if "hipaa" in normalized:
        return (
            "The indexed NYDOH standards may not use the word **HIPAA** in the "
            "retrieved passage. The answer below connects the question to the "
            "NYDOH requirements concerning patient-information confidentiality, "
            "privacy, authorized access, and disclosure. HIPAA applicability itself "
            "should be verified separately by the laboratory's compliance leadership."
        )

    if "phi" in normalized or "protected health information" in normalized:
        return (
            "The NYDOH standards may describe this concept as patient-information "
            "confidentiality, privacy, access, or disclosure rather than using the "
            "term **PHI**."
        )

    return None


def answer_question(query, vector_db, client, chunks, catalog, k=2):
    """
    Answer retrieval remains fixed at k=2.

    A wider candidate search is used only for related-standard discovery.
    """
    index_request = detect_index_request(
        query,
        sorted({item["prefix"] for item in catalog.values()}),
    )

    if index_request:
        return {
            "answer": format_standard_index(
                catalog,
                prefix=index_request["prefix"],
            ),
            "concept_note": None,
            "sources": [],
            "related_standards": [],
            "connections": None,
            "checklist": None,
            "web_ideas": None,
            "index_mode": True,
            "index_prefix": index_request["prefix"],
        }

    requested_standard = standard_search_query(query)

    if requested_standard:
        exact_docs = find_chunks_by_standard(chunks, requested_standard)
        docs = exact_docs[:k]

        if len(docs) < k:
            fallback = vector_db.similarity_search(query, k=k)
            existing_ids = {
                (doc.metadata.get("source"), doc.metadata.get("page"))
                for doc in docs
            }

            for doc in fallback:
                key = (doc.metadata.get("source"), doc.metadata.get("page"))
                if key not in existing_ids:
                    docs.append(doc)
                    existing_ids.add(key)
                if len(docs) >= k:
                    break
    else:
        retrieval_query = expand_query_for_retrieval(query)
        docs = vector_db.similarity_search(retrieval_query, k=k)

    if not docs:
        return {
            "answer": "I do not see a clear answer in the provided NYDOH materials.",
            "concept_note": get_concept_note(query),
            "sources": [],
            "related_standards": [],
            "connections": None,
            "checklist": None,
            "web_ideas": None,
            "index_mode": False,
            "index_prefix": None,
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
- Cite the exact standard code and title when they are identifiable.
- If the user uses a common external term such as HIPAA or PHI and that exact
  term is not present in the context, do not claim that NYDOH defines it.
  Explain which NYDOH concepts are related.
- Clearly distinguish an NYDOH requirement from a broader healthcare concept.

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

    grounded_answer = response.choices[0].message.content.strip()

    sources = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata
        standard = metadata.get("standard") or extract_standard(doc.page_content)
        standard = normalize_standard(str(standard))

        page = metadata.get("page_label")
        if page is None:
            raw_page = metadata.get("page")
            page = str(raw_page + 1) if isinstance(raw_page, int) else "Unknown"

        title = extract_standard_title(doc.page_content, standard)
        paragraph = best_supporting_paragraph(query, doc.page_content)

        source_name = metadata.get("source", "Unknown")
        pdf_url = GENERAL_URL if source_name == "NYDOH General" else HISTO_URL
        pdf_link = f"{pdf_url}#page={page}"

        item = {
            "standard": standard,
            "title": title,
            "source": source_name,
            "page": str(page),
            "pdf_link": pdf_link,
            "context": re.sub(r"\s+", " ", doc.page_content).strip(),
            "supporting_paragraph": paragraph,
            "highlighted_paragraph": highlight_query_terms(paragraph, query),
        }

        key = (item["standard"], item["source"], item["page"])
        if key not in seen:
            seen.add(key)
            sources.append(item)

    # Broader retrieval only for related-standard discovery.
    related_candidates = vector_db.similarity_search(
        expand_query_for_retrieval(query),
        k=6,
    )

    related_standards = []
    related_seen = {item["standard"] for item in sources}

    for doc in related_candidates:
        metadata = doc.metadata
        standard = normalize_standard(
            str(metadata.get("standard") or extract_standard(doc.page_content))
        )

        if standard == "NOT IDENTIFIED" or standard in related_seen:
            continue

        page = metadata.get("page_label")
        if page is None:
            raw_page = metadata.get("page")
            page = str(raw_page + 1) if isinstance(raw_page, int) else "Unknown"

        title = extract_standard_title(doc.page_content, standard)
        source_name = metadata.get("source", "Unknown")
        pdf_url = GENERAL_URL if source_name == "NYDOH General" else HISTO_URL

        related_standards.append(
            {
                "standard": standard,
                "title": title,
                "source": source_name,
                "page": str(page),
                "pdf_link": f"{pdf_url}#page={page}",
                "context": best_supporting_paragraph(query, doc.page_content),
            }
        )
        related_seen.add(standard)

        if len(related_standards) >= 4:
            break

    source_text = "\n\n".join(item["context"] for item in sources)

    checklist = build_compliance_checklist(
        query,
        grounded_answer,
        source_text,
        client,
    )

    connection_sources = sources + related_standards
    connections = explain_standard_connections(
        query,
        connection_sources,
        client,
    )

    concept_note = get_concept_note(query)
    web_ideas = None

    if asks_for_compliance_ideas(query):
        try:
            web_ideas = get_web_compliance_ideas(query, client)
        except Exception as exc:
            web_ideas = (
                "### Proposed implementation ideas\n\n"
                "A web search could not be completed at this time.\n\n"
                "**Disclaimer:** These are proposed solutions found on the internet, "
                "but they must be confirmed / verified by the Lab Director.\n\n"
                f"_Web-search error: {exc}_"
            )

    return {
        "answer": grounded_answer,
        "concept_note": concept_note,
        "sources": sources,
        "related_standards": related_standards,
        "connections": connections,
        "checklist": checklist,
        "web_ideas": web_ideas,
        "index_mode": False,
        "index_prefix": None,
    }

if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "history" not in st.session_state:
    st.session_state.history = []

api_key = get_openai_api_key()
(
    vector_db,
    client,
    page_count,
    chunk_count,
    all_chunks,
    standard_catalog,
) = initialize_rag(api_key)

available_prefixes = sorted(
    {item["prefix"] for item in standard_catalog.values()}
)

with st.sidebar:
    # -----------------------------------------------------
    # 1. STANDARDS INDEX — PRIMARY NAVIGATION
    # -----------------------------------------------------
    st.markdown("## 📚 Standards Index")
    st.caption("Browse or open NYDOH standards by section.")

    selected_section = st.selectbox(
        "Browse by section",
        ["Select a section"] + available_prefixes,
    )

    if selected_section != "Select a section":
        section_count = sum(
            1 for item in standard_catalog.values()
            if item["prefix"] == selected_section
        )

        st.caption(
            f"{section_count} standards found in {selected_section}."
        )

        if st.button(
            f"View all {selected_section} standards",
            use_container_width=True,
        ):
            index_result = {
                "answer": format_standard_index(
                    standard_catalog,
                    prefix=selected_section,
                ),
                "concept_note": None,
                "sources": [],
                "related_standards": [],
                "connections": None,
                "checklist": None,
                "web_ideas": None,
                "index_mode": True,
                "index_prefix": selected_section,
            }

            synthetic_question = (
                f"Show all {selected_section} standards"
            )

            st.session_state.current_question = synthetic_question
            st.session_state.current_result = index_result
            st.session_state.history.append(
                {
                    "question": synthetic_question,
                    "result": index_result,
                }
            )
            st.rerun()

    if st.button(
        "View full standards index",
        use_container_width=True,
    ):
        index_result = {
            "answer": format_standard_index(
                standard_catalog,
                prefix=None,
            ),
            "concept_note": None,
            "sources": [],
            "related_standards": [],
            "connections": None,
            "checklist": None,
            "web_ideas": None,
            "index_mode": True,
            "index_prefix": None,
        }

        st.session_state.current_question = "Show the standards index"
        st.session_state.current_result = index_result
        st.session_state.history.append(
            {
                "question": "Show the standards index",
                "result": index_result,
            }
        )
        st.rerun()

    st.markdown("---")

    # -----------------------------------------------------
    # 2. RECENT QUESTIONS
    # -----------------------------------------------------
    st.markdown("### 🕘 Recent Questions")

    if not st.session_state.history:
        st.caption("No questions yet.")
    else:
        for index, item in enumerate(reversed(st.session_state.history[-8:])):
            label = item["question"]

            if len(label) > 42:
                label = label[:42] + "…"

            if st.button(
                label,
                key=f"history_{index}",
                use_container_width=True,
            ):
                st.session_state.current_question = item["question"]
                st.session_state.current_result = item["result"]
                st.rerun()

        if st.button(
            "Clear history",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.session_state.current_result = None
            st.session_state.current_question = ""
            st.rerun()

    st.markdown("---")

    # -----------------------------------------------------
    # 3. PHI WARNING — PROMINENT
    # -----------------------------------------------------
    st.error(
        """
        **⚠️ IMPORTANT — DO NOT ENTER PHI**

        Never enter Protected Health Information (PHI), patient names,
        medical record numbers, dates of birth, or any other
        patient-identifiable information into this application.

        Use this assistant only for regulatory, compliance, policy,
        procedure, and hypothetical questions.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # 4. ABOUT
    # -----------------------------------------------------
    st.markdown("### ℹ️ About")

    st.markdown(
        """
        **NYDOH answers:** based only on the indexed standards.

        **Compliance ideas:** a public web search is used only when the
        question asks how to comply with or implement a standard.

        **Verification:** critical compliance decisions should always be
        confirmed against the official NYDOH documents and reviewed by
        the Laboratory Director.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # 5. SYSTEM INFORMATION — BOTTOM / COLLAPSED
    # -----------------------------------------------------
    with st.expander("🔧 System Information", expanded=False):
        st.markdown(
            f"""
            **Official Standards Loaded**  
            {page_count} pages

            **Indexed Knowledge Base**  
            {chunk_count} searchable sections

            **Evidence Retrieved**  
            2 passages

            **Retrieval Setting**  
            Fixed at `k = 2` for consistent behavior
            """
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
            result = answer_question(
                cleaned,
                vector_db,
                client,
                all_chunks,
                standard_catalog,
                2,
            )
        st.session_state.current_question = cleaned
        st.session_state.current_result = result
        st.session_state.history.append({"question": cleaned, "result": result})
        st.rerun()

st.markdown("---")
left, right = st.columns([1.7, 1])

with left:
    st.markdown('<div class="eyebrow">Current answer</div>', unsafe_allow_html=True)

    if st.session_state.current_result is None:
        with st.container(border=True):
            st.markdown(
                '<div class="empty-state">Enter a question above to begin.</div>',
                unsafe_allow_html=True,
            )
    else:
        result = st.session_state.current_result

        if result.get("index_mode"):
            with st.container(border=True):
                st.markdown(result["answer"])
        else:
            answer_tab, checklist_tab, connections_tab = st.tabs(
                ["Answer", "Compliance checklist", "Connected standards"]
            )

            with answer_tab:
                with st.container(border=True):
                    concept_note = result.get("concept_note")
                    if concept_note:
                        st.info(concept_note)

                    st.markdown(result["answer"])

                    web_ideas = result.get("web_ideas")
                    if web_ideas:
                        st.markdown("---")
                        st.markdown(web_ideas)

            with checklist_tab:
                with st.container(border=True):
                    checklist = result.get("checklist")
                    if checklist:
                        st.markdown(checklist)
                        st.caption(
                            "This checklist is derived only from the retrieved NYDOH text "
                            "and should be reviewed against the full standard."
                        )
                    else:
                        st.info("A checklist could not be generated from the retrieved text.")

            with connections_tab:
                with st.container(border=True):
                    connections = result.get("connections")
                    if connections:
                        st.markdown(connections)
                    else:
                        st.info("No additional standard connections were identified.")

with right:
    st.markdown('<div class="eyebrow">Supporting standards</div>', unsafe_allow_html=True)
    if st.session_state.current_result is None:
        st.markdown(
            '<div class="panel"><div class="empty-state">Sources will appear here.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        current_result = st.session_state.current_result

        if current_result.get("index_mode"):
            selected_prefix = current_result.get("index_prefix")

            if selected_prefix:
                count = sum(
                    1 for item in standard_catalog.values()
                    if item["prefix"] == selected_prefix
                )
                st.success(
                    f"{count} standards listed in {selected_prefix}."
                )
            else:
                st.success(
                    f"{len(standard_catalog)} standards listed across "
                    f"{len(available_prefixes)} sections."
                )

            st.markdown("**Available sections**")
            st.write(", ".join(available_prefixes))
        else:
            sources = current_result["sources"]

            if not sources:
                st.info("No supporting passages were retrieved.")
            else:
                for source in sources:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-title">
                                {source['standard']} — {source['title']}
                            </div>
                            <div><strong>Page {source['page']}</strong> ·
                            <a href="{source['pdf_link']}" target="_blank">Open PDF</a></div>
                            <div class="source-text" style="margin-top:.55rem;">
                                {source['highlighted_paragraph']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                related = current_result.get("related_standards", [])
                if related:
                    st.markdown("#### Related standards")
                    for item in related:
                        st.markdown(
                            f"- **[{item['standard']} — {item['title']}]"
                            f"({item['pdf_link']})**, page {item['page']}"
                        )
