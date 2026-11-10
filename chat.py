#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Streamlit chat interface that ties together:
- rag_client      (ChromaDB discovery, semantic retrieval, context formatting)
- llm_client      (OpenAI answer generation with conversation history)
- ragas_evaluator (real-time answer quality scores)
"""

import os
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

import llm_client
import rag_client
import ragas_evaluator

load_dotenv()

# RAGAS imports (ragas_evaluator applies the import compatibility shim first)
try:
    from ragas import SingleTurnSample  # noqa: F401
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

if not RAGAS_AVAILABLE:
    st.warning("RAGAS not available. Install with: pip install ragas")

MISSION_OPTIONS = {
    "all": "All missions",
    "apollo11": "Apollo 11",
    "apollo13": "Apollo 13",
    "challenger": "Challenger (STS-51L)",
}


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    return rag_client.discover_chroma_backends()


@st.cache_resource
def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None


def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context"""
    return rag_client.format_context(documents, metadatas, distances)


def generate_response(openai_key, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-4.1-mini") -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(openai_key, user_message, context, conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              openai_key: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts, openai_key=openai_key)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")
    st.sidebar.caption("RAGAS scores for the last answer (1.0 = best)")

    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{score:.3f}",
                delta=None
            )
            st.sidebar.progress(float(min(max(score, 0.0), 1.0)))

    for warning in scores.get("warnings", []) or []:
        st.sidebar.warning(warning)


def display_sources(docs_result: Optional[Dict]):
    """Show the retrieved chunks under the answer for transparency."""
    if not docs_result or not docs_result.get("documents") or not docs_result["documents"][0]:
        st.caption("No documents were retrieved for this question.")
        return
    documents = docs_result["documents"][0]
    metadatas = docs_result["metadatas"][0]
    distances = (docs_result.get("distances") or [[None] * len(documents)])[0]
    with st.expander(f"📄 Retrieved sources ({len(documents)})"):
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            meta = meta or {}
            mission = str(meta.get("mission", "unknown")).replace("_", " ").title()
            source = meta.get("source", "unknown")
            category = str(meta.get("document_category", "general")).replace("_", " ").title()
            dist_text = f" · distance {dist:.3f}" if isinstance(dist, (int, float)) else ""
            st.markdown(f"**{i}. {mission} · {source} · {category}**{dist_text}")
            st.text(doc[:600] + ("..." if len(doc) > 600 else ""))


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    if "last_docs_result" not in st.session_state:
        st.session_state.last_docs_result = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # Discover available backends
        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n"
                    "`python embedding_pipeline.py --openai-key YOUR_KEY --data-path ./data_text`")
            st.stop()

        # Backend selection
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}

        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )

        selected_backend = available_backends[selected_backend_key]

        # API Key input
        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )

        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        else:
            # Used by the query embedding (rag_client) and the RAGAS evaluator (langchain-openai)
            os.environ["CHROMA_OPENAI_API_KEY"] = openai_key
            os.environ["OPENAI_API_KEY"] = openai_key

        # Model selection
        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
            help="Choose the OpenAI model for responses"
        )

        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)
        mission_choice = st.selectbox(
            "Mission focus",
            options=list(MISSION_OPTIONS.keys()),
            format_func=lambda x: MISSION_OPTIONS[x],
            help="Restrict retrieval to a single mission via metadata filtering"
        )

        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE)

        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.session_state.last_contexts = []
            st.session_state.last_docs_result = None
            st.rerun()

        # Initialize RAG system when backend changes
        if st.session_state.current_backend != selected_backend_key:
            st.session_state.current_backend = selected_backend_key
            # Clear cache to force reinitialization
            st.cache_resource.clear()

    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):
        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )

    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    # Display evaluation metrics if available
    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sources of the last answer
    if st.session_state.messages and st.session_state.last_docs_result:
        display_sources(st.session_state.last_docs_result)

    # Chat input
    if prompt := st.chat_input("Ask about NASA space missions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Retrieve relevant documents (optionally restricted to one mission)
                docs_result = retrieve_documents(
                    collection,
                    prompt,
                    n_docs,
                    mission_choice
                )
                st.session_state.last_docs_result = docs_result

                # Format context
                context = ""
                contexts_list: List[str] = []
                if docs_result and docs_result.get("documents"):
                    documents = docs_result["documents"][0]
                    metadatas = docs_result["metadatas"][0]
                    distances = (docs_result.get("distances") or [None])[0]
                    context = format_context(documents, metadatas, distances)
                    contexts_list = rag_client.unique_documents(documents)
                    st.session_state.last_contexts = contexts_list

                # Generate response
                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model_choice
                )
                st.markdown(response)

                # Evaluate response quality if enabled
                if enable_evaluation and RAGAS_AVAILABLE:
                    with st.spinner("Evaluating response quality..."):
                        evaluation_scores = evaluate_response_quality(
                            prompt,
                            response,
                            contexts_list,
                            openai_key
                        )
                        st.session_state.last_evaluation = evaluation_scores

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
